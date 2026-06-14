"""Build the judge-focused QuantSafe demo from verified production captures."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
DEFAULT_CAPTURE_DIR = ROOT / "output" / "playwright" / "demo-capture"
BUILD_DIR = ROOT / "output" / "demo-build"

W, H = 1280, 720
IVORY = (250, 249, 246)
INK = (26, 26, 26)
WARM = (74, 69, 62)
OXBLOOD = (123, 45, 38)
GOLD = (201, 162, 75)
SAGE = (79, 111, 82)
TRANSITION = 0.35
FPS = 30

FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
FONT_SERIF = Path(r"C:\Windows\Fonts\georgiab.ttf")
FONT_ITALIC = Path(r"C:\Windows\Fonts\georgiai.ttf")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def fit_text(draw: ImageDraw.ImageDraw, text: str, face, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=face) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def add_chapter(draw: ImageDraw.ImageDraw, label: str) -> None:
    face = font(FONT_BOLD, 16)
    width = int(draw.textlength(label, font=face)) + 28
    draw.rounded_rectangle((24, 22, 24 + width, 56), radius=8, fill=OXBLOOD)
    draw.text((38, 30), label, font=face, fill=IVORY)


def screenshot_slide(
    source: Path,
    chapter: str,
    headline: str,
    detail: str,
) -> Image.Image:
    image = Image.open(source).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    add_chapter(draw, chapter)

    panel_top = 584
    draw.rectangle((0, panel_top, W, H), fill=(250, 249, 246, 246))
    draw.rectangle((0, panel_top, W, panel_top + 3), fill=GOLD)
    draw.text((34, 603), headline, font=font(FONT_BOLD, 25), fill=OXBLOOD)

    detail_face = font(FONT_REGULAR, 18)
    lines = fit_text(draw, detail, detail_face, W - 68)
    y = 642
    for line in lines[:2]:
        draw.text((34, y), line, font=detail_face, fill=WARM)
        y += 24
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def title_slide() -> Image.Image:
    image = Image.new("RGB", (W, H), IVORY)
    draw = ImageDraw.Draw(image)
    draw.text((W // 2, 62), "QUANTIZATION SAFETY SCREEN",
              font=font(FONT_BOLD, 15), fill=OXBLOOD, anchor="ma")
    draw.text((W // 2, 104), "QuantSafe", font=font(FONT_SERIF, 58),
              fill=INK, anchor="ma")
    draw.text((W // 2, 170), "will this quant jailbreak your model?",
              font=font(FONT_ITALIC, 25), fill=OXBLOOD, anchor="ma")
    draw.rectangle((W // 2 - 36, 211, W // 2 + 36, 213), fill=GOLD)

    draw.text((W // 2, 282), "One of mine did.",
              font=font(FONT_BOLD, 34), fill=INK, anchor="ma")
    draw.text((W // 2, 348), "91%  ->  1%",
              font=font(FONT_SERIF, 76), fill=OXBLOOD, anchor="ma")
    draw.text((W // 2, 444),
              "Quantization kept the benchmarks - and erased the refusals.",
              font=font(FONT_REGULAR, 24), fill=WARM, anchor="ma")

    draw.rounded_rectangle((284, 518, 996, 588), radius=12,
                           fill=(243, 239, 233), outline=(216, 210, 199), width=2)
    draw.text((W // 2, 540),
              "A live release gate: detect -> route -> sign -> verify -> escalate",
              font=font(FONT_BOLD, 20), fill=INK, anchor="ma")
    draw.text((W // 2, 655), "Production Space | 49-second judge cut",
              font=font(FONT_REGULAR, 17), fill=WARM, anchor="ma")
    return image


def close_slide() -> Image.Image:
    image = Image.new("RGB", (W, H), IVORY)
    draw = ImageDraw.Draw(image)
    draw.text((W // 2, 84), "QuantSafe Certifier",
              font=font(FONT_SERIF, 54), fill=INK, anchor="ma")
    draw.text((W // 2, 158), "Small models. Verifiable release decisions.",
              font=font(FONT_ITALIC, 26), fill=OXBLOOD, anchor="ma")
    draw.rectangle((W // 2 - 36, 204, W // 2 + 36, 206), fill=GOLD)

    rows = [
        ("45", "measured quantization cells"),
        ("34", "GGUF cells through llama.cpp / Ollama"),
        ("0.84", "family-transfer ROC AUC"),
        ("97.73%", "held-out accuracy from the ModernBERT cross-check"),
    ]
    y = 255
    for metric, label in rows:
        draw.text((300, y), metric, font=font(FONT_SERIF, 36),
                  fill=OXBLOOD, anchor="ra")
        draw.text((330, y + 5), label, font=font(FONT_REGULAR, 21), fill=INK)
        y += 62

    draw.rounded_rectangle((206, 512, 1074, 590), radius=12,
                           fill=(236, 240, 234), outline=SAGE, width=2)
    draw.text((W // 2, 533),
              "ZeroGPU | Modal | Gradio | Ed25519 | arXiv:2606.10154",
              font=font(FONT_BOLD, 20), fill=(54, 75, 56), anchor="ma")
    draw.text((W // 2, 627),
              "huggingface.co/spaces/build-small-hackathon/quantsafe-certifier",
              font=font(FONT_BOLD, 18), fill=OXBLOOD, anchor="ma")
    draw.text((W // 2, 666), "Open source | Reproducible | Under the 32B limit",
              font=font(FONT_REGULAR, 16), fill=WARM, anchor="ma")
    return image


SLIDES = [
    ("00-title.png", 4.20, None, "", ""),
    (
        "01-failure.png", 4.80, "01 / FAILURE",
        "Benchmarks stayed flat. Refusals collapsed.",
        "The published phi-2 GPTQ release fell from 91% refusal to 1%.",
    ),
    (
        "02-route.png", 5.40, "02 / DETECT + ROUTE",
        "QuantSafe calls HIGH - and blocks the release.",
        "RTSI 0.6199. Route the riskiest 20% and recover 76% of the refusal-rate gap.",
    ),
    (
        "10-zerogpu.png", 4.80, "03 / LIVE ZEROGPU",
        "A real RTX Pro 6000 probe, not a mock.",
        "Two Qwen checkpoints, ten private probes, aggregate drift only. Completed in 27 seconds.",
    ),
    (
        "03-certificate.png", 5.30, "04 / BIND",
        "Turn the decision into a portable signed record.",
        "The record binds the action to a published Hub revision, evidence hashes, and issuer identity.",
    ),
    (
        "04-verified.png", 4.40, "05 / VERIFY",
        "The production issuer key verifies.",
        "Ed25519 verification is pinned to the README-published Space key - not the key inside the record.",
    ),
    (
        "05-tampered.png", 4.40, "06 / ATTACK",
        "Flip one signed field: INVALID.",
        "Tampering breaks the signature; a foreign re-sign is rejected as issuer substitution.",
    ),
    (
        "06-debate-top.png", 4.80, "07 / ESCALATE",
        "Borderline calls go to a constitutional debate.",
        "Qwen3-8B, Phi-4-mini, and SmolLM3 argue independently through authenticated Modal workers.",
    ),
    (
        "08-debate-verdict.png", 5.00, "08 / CONSENSUS",
        "A genuine two-thirds decision - not a single-model guess.",
        "The final round reaches CONDITIONAL at 67% agreement and exposes the dissenting ROUTE vote.",
    ),
    (
        "09-evidence.png", 4.60, "09 / EVIDENCE",
        "Research-backed, scoped, and explicit about limits.",
        "45 cells, family-transfer validation, a fine-tuned semantic cross-check, and arXiv:2606.10154.",
    ),
    ("10-close.png", 5.20, None, "", ""),
]


def build_slides(capture_dir: Path) -> list[Path]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, _duration, chapter, headline, detail in SLIDES:
        target = BUILD_DIR / name
        if name == "00-title.png":
            image = title_slide()
        elif name == "10-close.png":
            image = close_slide()
        else:
            source = capture_dir / name
            if not source.exists():
                raise FileNotFoundError(f"missing verified capture: {source}")
            image = screenshot_slide(source, chapter or "", headline, detail)
        image.save(target, optimize=True)
        paths.append(target)
    return paths


def video_filter() -> str:
    chains = []
    for index, (_name, duration, *_rest) in enumerate(SLIDES):
        chains.append(
            f"[{index}:v]fps={FPS},format=yuv420p,trim=duration={duration},"
            f"settb=AVTB,setpts=PTS-STARTPTS[v{index}]"
        )

    current = "v0"
    total = SLIDES[0][1]
    for index in range(1, len(SLIDES)):
        offset = total - TRANSITION
        output = f"x{index}"
        chains.append(
            f"[{current}][v{index}]xfade=transition=fade:"
            f"duration={TRANSITION}:offset={offset:.2f}[{output}]"
        )
        total += SLIDES[index][1] - TRANSITION
        current = output
    chains.append(f"[{current}]format=yuv420p[out]")
    return ";".join(chains)


def run_ffmpeg(slide_paths: list[Path]) -> None:
    inputs: list[str] = []
    for path, (_name, duration, *_rest) in zip(slide_paths, SLIDES):
        inputs.extend(["-loop", "1", "-t", str(duration), "-i", str(path)])

    mp4 = DEMO_DIR / "quantsafe-demo.mp4"
    webm = DEMO_DIR / "quantsafe-demo.webm"
    mp4_tmp = BUILD_DIR / "quantsafe-demo.new.mp4"
    webm_tmp = BUILD_DIR / "quantsafe-demo.new.webm"
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
            "-filter_complex", video_filter(), "-map", "[out]", "-r", str(FPS)]
    subprocess.run(
        [*base, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-movflags", "+faststart", str(mp4_tmp)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp4_tmp),
         "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "28", "-row-mt", "1",
         "-pix_fmt", "yuv420p", str(webm_tmp)],
        check=True,
    )
    for source, target in ((mp4_tmp, mp4), (webm_tmp, webm)):
        for attempt in range(5):
            try:
                source.replace(target)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.25)


def write_captions() -> None:
    def stamp(seconds: float) -> str:
        millis = round(seconds * 1000)
        hours, millis = divmod(millis, 3_600_000)
        minutes, millis = divmod(millis, 60_000)
        secs, millis = divmod(millis, 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    entries = []
    start = 0.0
    for index, (_name, duration, _chapter, headline, detail) in enumerate(SLIDES, 1):
        end = start + duration
        if index == 1:
            text = "One of my quantized releases kept its benchmarks and lost its refusals."
        elif index == len(SLIDES):
            text = "QuantSafe: small models, verifiable release decisions."
        else:
            text = f"{headline}\n{detail}"
        entries.append(f"{index}\n{stamp(start)} --> {stamp(end)}\n{text}\n")
        start = end - TRANSITION
    (DEMO_DIR / "captions.srt").write_text("\n".join(entries), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    args = parser.parse_args()
    paths = build_slides(args.capture_dir.resolve())
    run_ffmpeg(paths)
    write_captions()
    print(f"Built {DEMO_DIR / 'quantsafe-demo.mp4'}")
    print(f"Built {DEMO_DIR / 'quantsafe-demo.webm'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
