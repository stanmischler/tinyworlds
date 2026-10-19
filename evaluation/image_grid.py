"""Pixel-exact image grid: frames scaled up by an integer factor with nearest neighbour (no resampling blur), a header
band per column, a caption band under each cell and a label band left of each row."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG, FG = (255, 255, 255), (20, 20, 20)
PAD = 4  # pixels between cells


def to_pil(frame, scale):
    # float [C, H, W] in [0, 1] -> PIL RGB [H*scale, W*scale], nearest neighbour
    arr = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # [H, W, C]
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def wrap(text, font, width, draw):
    # greedy word wrap so a column header fits its cell
    lines, line = [], ''
    for word in text.split():
        cand = f'{line} {word}'.strip()
        if line and draw.textlength(cand, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = cand
    return lines + [line]


def render_grid(rows, headers, row_labels, captions, scale=2, font_size=14):
    """rows: list of rows, each a list of frames float [C, H, W] in [0, 1]; headers: one string per column;
    row_labels: one string per row; captions: per row, one string per column ('' for none) -> PIL image."""
    font = ImageFont.load_default(size=font_size)
    line_h = font_size + 4
    cell_w, cell_h = rows[0][0].shape[2] * scale, rows[0][0].shape[1] * scale
    scratch = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    header_lines = [wrap(h, font, cell_w - PAD, scratch) for h in headers]
    header_h = line_h * max(len(lines) for lines in header_lines) + PAD
    label_w = max(int(scratch.textlength(line, font=font)) for lab in row_labels for line in lab.split('\n')) + 2 * PAD
    row_h = cell_h + line_h + PAD  # cell + caption
    n_cols = len(headers)
    canvas = Image.new('RGB', (label_w + n_cols * (cell_w + PAD), header_h + len(rows) * row_h), BG)
    draw = ImageDraw.Draw(canvas)
    for c, lines in enumerate(header_lines):
        x0 = label_w + c * (cell_w + PAD)
        for k, line in enumerate(lines):
            draw.text((x0 + cell_w / 2, k * line_h), line, fill=FG, font=font, anchor='ma')
    for r, frames in enumerate(rows):
        y0 = header_h + r * row_h
        draw.multiline_text((PAD, y0 + cell_h / 2), row_labels[r], fill=FG, font=font, anchor='lm')
        for c, frame in enumerate(frames):
            x0 = label_w + c * (cell_w + PAD)
            canvas.paste(to_pil(frame, scale), (x0, y0))
            draw.text((x0 + cell_w / 2, y0 + cell_h + 2), captions[r][c], fill=FG, font=font, anchor='ma')
    return canvas
