import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import statistics
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


GiB = 1024 ** 3
SIDES = ("image_left_eye", "image_right_eye")
STATES = ("open", "closed", "narrow", "occluded")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/workspace/dms-eye-status-turbo/qwen_models/Qwen3.5-9B",
    )
    parser.add_argument("--image", default="/workspace/dms-eye-status-turbo/extract_data/FA-The-67th-batch-delivery-20260116/E22_DMS/BlinkDetection/dms/pc1_20251216_2id/2025121602_20251216-161319_CmpRol/APillar_E22_seat1_2025121602_m_174_40_Beige_ComplexRollPose_NoOcclusion_c1lR1_StaticOutdoor_SunnyNormal_1340_NormalScene_000002.jpg")
    parser.add_argument("--gpu", default="0", help="物理GPU编号，仅使用这一张")
    parser.add_argument(
        "--out",
        default="/workspace/dms-eye-status-turbo/vlm_eye_experiment/runs/s01",
    )
    parser.add_argument("--context-budget", type=int, default=5600)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=3, help="首次之后的推理次数")
    return parser.parse_args()


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def memory():
    torch.cuda.synchronize(0)
    free, total = torch.cuda.mem_get_info(0)
    return {
        "allocated_GiB": torch.cuda.memory_allocated(0) / GiB,
        "reserved_GiB": torch.cuda.memory_reserved(0) / GiB,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(0) / GiB,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(0) / GiB,
        "device_used_GiB": (total - free) / GiB,
        "device_free_GiB": free / GiB,
    }


def build_instruction(width, height):
    return (
        "Locate the driver's two eyes in this image "
        f"(original width={width}, height={height}). "
        "image_left_eye is the eye with the smaller horizontal box-center "
        "coordinate; image_right_eye has the larger coordinate. "
        "Use image positions, not anatomical left and right. "
        "For each eye, output bbox and state. "
        "bbox must be [xmin,ymin,xmax,ymax], using integer pixel coordinates "
        "relative to the original image, with the origin at the top-left. "
        "state must be open, closed, narrow, or occluded. "
        "Use occluded when lighting, hair, or other factors prevent reliable "
        "classification as open, closed, or narrow. "
        "Return exactly one compact JSON object in exactly this form: "
        '{"image_left_eye":{"bbox":[100,200,130,220],"state":"open"},'
        '"image_right_eye":{"bbox":[150,200,180,220],"state":"closed"}}. '
        "Keep the JSON keys and nesting exactly unchanged; replace only "
        "the bbox numbers and state values. "
        "Do not use a JSON array, 'bbox_2d', or 'label'. "
        "Do not include explanations, markdown, or any additional fields."
    )

def normalize_json_text(raw_output):
    """只允许裸JSON，或完整包裹JSON的单个代码围栏。"""
    text = raw_output.strip()

    if not text.startswith("```"):
        return text

    lines = text.splitlines()

    if (
        len(lines) < 3
        or lines[0].strip().lower() not in ("```json", "```")
        or lines[-1].strip() != "```"
    ):
        raise ValueError("代码围栏不完整或类型不是JSON")

    body = "\n".join(lines[1:-1]).strip()

    if not body:
        raise ValueError("代码围栏内没有JSON内容")

    return body


def check_prediction(raw_output, width, height):
    errors = []

    # 拒绝重复字段与非标准JSON数值，避免解析时静默覆盖
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复JSON字段: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"非法JSON常量: {value}")

    try:
        normalized_output = normalize_json_text(raw_output)
        prediction = json.loads(
            normalized_output,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except ValueError as exc:
        return None, [f"invalid_json: {exc}"]

    if not isinstance(prediction, dict) or set(prediction) != set(SIDES):
        return prediction, ["顶层字段不符合约定"]

    valid_boxes = {}
    for side in SIDES:
        eye = prediction[side]
        if not isinstance(eye, dict) or set(eye) != {"bbox", "state"}:
            errors.append(f"{side}: 字段不符合约定")
            continue

        if eye["state"] not in STATES:
            errors.append(f"{side}: 非法state")

        box = eye["bbox"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(value) is int for value in box)
        ):
            errors.append(f"{side}: bbox必须为4个整数")
            continue

        x1, y1, x2, y2 = box
        # 图像边界口径：右/下边界允许等于 width/height
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors.append(f"{side}: bbox越界或面积非正")
            continue
        valid_boxes[side] = box

    if len(valid_boxes) == 2:
        left = valid_boxes[SIDES[0]]
        right = valid_boxes[SIDES[1]]
        if left[0] + left[2] >= right[0] + right[2]:
            errors.append("左右框中心x顺序错误或相同")

    return prediction, errors


def run_once(model, processor, gpu_inputs, width, height, max_new_tokens, name):
    # 输入已在GPU；计时不包含预处理、输入搬运、文本解码
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    before = memory()

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **gpu_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            return_dict_in_generate=False,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - start
    after = memory()

    input_length = gpu_inputs["input_ids"].shape[-1]
    new_ids = output_ids[0, input_length:].cpu().tolist()
    raw_output = processor.tokenizer.decode(
        new_ids, skip_special_tokens=True
    )
    output_with_special_tokens = processor.tokenizer.decode(
        new_ids, skip_special_tokens=False
    )

    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = processor.tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    ended_with_eos = bool(new_ids) and new_ids[-1] in (eos_ids or [])
    prediction, errors = check_prediction(raw_output, width, height)

    result = {
        "run": name,
        "generate_seconds": elapsed,
        "input_tokens": input_length,
        "generated_tokens_including_special": len(new_ids),
        "max_new_tokens": max_new_tokens,
        "ended_with_eos": ended_with_eos,
        "hit_token_limit": len(new_ids) >= max_new_tokens,
        "before": before,
        "after": after,
        "peak_allocated_above_baseline_GiB": (
            after["peak_allocated_GiB"] - before["allocated_GiB"]
        ),
        "raw_output": raw_output,
        "output_with_special_tokens": output_with_special_tokens,
        "prediction": prediction,
        "schema_and_geometry_errors": errors,
        "schema_and_geometry_pass": not errors,
    }

    # 不将GPU输出带到下一轮，不清空allocator缓存
    del output_ids
    return result


def main():
    args = parse_args()
    if args.repeat < 0:
        raise ValueError("--repeat不能小于0")
    if args.max_new_tokens <= 0 or args.context_budget <= 0:
        raise ValueError("token预算必须大于0")

    # 此时尚未进行任何CUDA调用
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    image_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    out = Path(args.out) / datetime.now().strftime("infer_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print("结果目录:", out, flush=True)

    try:
        # 1. CPU预处理；统一EXIF方向后，宽高与坐标使用同一张图
        with Image.open(image_path) as source:
            source_size = list(source.size)
            exif_orientation = source.getexif().get(274)
            image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size

        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        instruction = build_instruction(width, height)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        grids = inputs["image_grid_thw"].tolist()
        ip = processor.image_processor
        patch = int(ip.patch_size)
        merge = int(ip.merge_size)
        visual_tokens = sum(
            int(t * h * w) // (merge * merge) for t, h, w in grids
        )
        image_token = getattr(processor, "image_token", "<|image_pad|>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
        actual_image_tokens = int(
            (inputs["input_ids"] == image_token_id).sum().item()
        )
        if actual_image_tokens != visual_tokens:
            raise ValueError(
                f"视觉token不一致: grid={visual_tokens}, "
                f"input={actual_image_tokens}"
            )

        input_tokens = inputs["input_ids"].shape[-1]
        active_tokens = int(inputs["attention_mask"].sum().item())
        required_context = input_tokens + args.max_new_tokens

        preparation = {
            "model": str(model_path),
            "image": str(image_path),
            "source_size_wh": source_size,
            "source_exif_orientation": exif_orientation,
            "input_image_size_wh": [width, height],
            "coordinate_reference": "EXIF方向统一后的整图；整数像素xyxy",
            "bbox_validation": "0 <= xmin < xmax <= width; 同理height",
            "instruction": instruction,
            "enable_thinking": False,
            "image_grid_thw": grids,
            "patch_size": patch,
            "merge_size": merge,
            "visual_tokens": visual_tokens,
            "actual_image_tokens": actual_image_tokens,
            "input_tokens": input_tokens,
            "active_tokens": active_tokens,
            "non_visual_tokens": input_tokens - visual_tokens,
            "context_budget": args.context_budget,
            "max_new_tokens": args.max_new_tokens,
            "required_context": required_context,
            "input_tensor_layout": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in inputs.items()
            },
        }
        write_json(out / "preparation.json", preparation)
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")
        print(json.dumps(preparation, ensure_ascii=False, indent=2), flush=True)

        if required_context > args.context_budget:
            raise ValueError(
                f"需要{required_context} tokens，超过实验预算"
                f"{args.context_budget}；未截断输入，未加载模型"
            )

        # 2. BF16模型加载及显存基线
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        torch.cuda.set_device(0)
        torch.cuda.init()
        torch.manual_seed(0)

        torch.cuda.reset_peak_memory_stats(0)
        before_load = memory()
        start = time.perf_counter()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        model.eval()
        torch.cuda.synchronize(0)
        load_seconds = time.perf_counter() - start
        after_load = memory()

        layout = {}
        for parameter in model.parameters():
            key = f"{parameter.device}/{parameter.dtype}"
            item = layout.setdefault(key, {"numel": 0, "bytes": 0})
            item["numel"] += parameter.numel()
            item["bytes"] += parameter.numel() * parameter.element_size()

        load_report = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": args.gpu,
            "gpu": torch.cuda.get_device_name(0),
            "model_class": type(model).__name__,
            "load_seconds": load_seconds,
            "before": before_load,
            "after": after_load,
            "parameter_layout": layout,
            "generation_config": model.generation_config.to_dict(),
        }
        write_json(out / "load_metrics.json", load_report)
        print("模型加载完成，开始单图推理", flush=True)

        # 3. 输入搬运；保留整数ID、mask和grid的dtype
        gpu_inputs = {
            key: value.to(
                device="cuda:0",
                dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
            )
            for key, value in inputs.items()
        }
        torch.cuda.synchronize(0)
        write_json(out / "input_on_gpu_memory.json", memory())

        # 4. 首次 + 后续重复；不记录memory history，避免其计时开销
        results = []
        for index in range(args.repeat + 1):
            name = "first" if index == 0 else f"repeat_{index:02d}"
            result = run_once(
                model, processor, gpu_inputs,
                width, height, args.max_new_tokens, name,
            )
            results.append(result)
            write_json(out / f"{name}_metrics.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

        # 5. 小样本测量汇总，不当作生产性能基准
        repeats = results[1:]
        summary = {
            "status": "completed",
            "timing_scope": (
                "同步计时的model.generate；不含预处理、输入搬运、"
                "解码、校验和文件写入；未启用memory history"
            ),
            "input_tokens": input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "first_seconds": results[0]["generate_seconds"],
            "repeat_count": len(repeats),
            "repeat_mean_seconds": (
                statistics.mean(r["generate_seconds"] for r in repeats)
                if repeats else None
            ),
            "repeat_median_seconds": (
                statistics.median(r["generate_seconds"] for r in repeats)
                if repeats else None
            ),
            "max_peak_allocated_GiB": max(
                r["after"]["peak_allocated_GiB"] for r in results
            ),
            "max_peak_reserved_GiB": max(
                r["after"]["peak_reserved_GiB"] for r in results
            ),
            "all_outputs_identical": all(
                r["raw_output"] == results[0]["raw_output"] for r in results
            ),
            "all_schema_and_geometry_pass": all(
                r["schema_and_geometry_pass"] for r in results
            ),
            "all_ended_with_eos": all(r["ended_with_eos"] for r in results),
            "quality_evaluated": False,
        }
        write_json(out / "summary.json", summary)
        print("汇总:", json.dumps(summary, ensure_ascii=False, indent=2))
        print("结果目录:", out)

    except Exception:
        failure = {
            "status": "failed",
            "traceback": traceback.format_exc(),
        }
        # 失败时尽力保存显存；CUDA错误可能导致此调用也失败
        try:
            if torch.cuda.is_initialized():
                failure["memory"] = memory()
        except Exception as memory_error:
            failure["memory_error"] = repr(memory_error)
        write_json(out / "error.json", failure)
        raise


if __name__ == "__main__":
    main()