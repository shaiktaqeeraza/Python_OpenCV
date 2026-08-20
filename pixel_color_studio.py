import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import cv2
import numpy as np
from PIL import Image, ImageTk


# --------------------------------------------------------------------------- #
# Theme palette (dark navy / purple, flat -- no gradients or glow)
# --------------------------------------------------------------------------- #
COLOR_BG          = "#0d0f1e"   # main window background
COLOR_PANEL       = "#141733"   # side panel / card background
COLOR_PANEL_ALT   = "#1b1e42"   # slightly lighter card
COLOR_ACCENT      = "#3fa9f5"   # cyan-blue accent (buttons, highlights)
COLOR_ACCENT_2    = "#7b2ff7"   # purple accent (secondary highlights)
COLOR_TEXT        = "#e8eaf6"   # light text
COLOR_TEXT_MUTED  = "#8b8fb8"   # muted/secondary text
COLOR_CANVAS_BG   = "#0a0c18"   # image preview background


# --------------------------------------------------------------------------- #
# Image processing helpers
# --------------------------------------------------------------------------- #
def apply_pixelate(img, block_size):
    if block_size <= 1:
        return img
    h, w = img.shape[:2]
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_posterize(img, levels):
    if levels >= 256:
        return img
    levels = max(2, levels)
    step = 256 / levels
    quantized = (np.floor(img.astype(np.float32) / step) * step + step / 2)
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    return quantized


def apply_color_replace(img, target_bgr, replacement_bgr, tolerance):
    """Replace pixels close to target_bgr (within tolerance) with replacement_bgr."""
    if target_bgr is None:
        return img
    diff = img.astype(np.int32) - np.array(target_bgr, dtype=np.int32)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = dist <= tolerance
    result = img.copy()
    result[mask] = replacement_bgr
    return result


def bgr_to_hex(bgr):
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


# --------------------------------------------------------------------------- #
# Reusable flat-styled widgets
# --------------------------------------------------------------------------- #
def make_flat_button(parent, text, command, accent=COLOR_ACCENT, fg="#ffffff"):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=accent,
        fg=fg,
        activebackground=COLOR_ACCENT_2,
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        padx=14,
        pady=6,
        font=("Segoe UI", 9, "bold"),
        cursor="hand2",
    )


# --------------------------------------------------------------------------- #
# Main Application
# --------------------------------------------------------------------------- #
class PixelColorStudio:
    DISPLAY_MAX_W = 720
    DISPLAY_MAX_H = 620

    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Color Studio")
        self.root.geometry("1180x740")
        self.root.minsize(950, 620)
        self.root.configure(bg=COLOR_BG)

        self._configure_ttk_style()

        self.original_image = None   # BGR numpy array, untouched
        self.processed_image = None  # BGR numpy array, after edits
        self.display_photo = None

        # Rendering metadata needed to map canvas clicks back to image pixels
        self._render_scale = 1.0
        self._render_offset = (0, 0)
        self._render_size = (0, 0)

        self.picked_color_bgr = None       # color picked from the image
        self.replacement_color_bgr = (0, 200, 255)  # default replacement (orange-ish in BGR)

        self._build_toolbar()
        self._build_body()

    # ----------------------------------------------------------------- #
    # ttk theme setup
    # ----------------------------------------------------------------- #
    def _configure_ttk_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 9))
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 9))
        style.configure(
            "Heading.TLabel",
            background=COLOR_PANEL,
            foreground=COLOR_ACCENT,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Muted.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT_MUTED, font=("Segoe UI", 8)
        )
        style.configure(
            "TCheckbutton",
            background=COLOR_PANEL,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 9),
        )
        style.map(
            "TCheckbutton",
            background=[("active", COLOR_PANEL)],
            foreground=[("active", COLOR_ACCENT)],
        )
        style.configure(
            "TScale",
            background=COLOR_PANEL,
            troughcolor=COLOR_PANEL_ALT,
        )
        style.configure("Vertical.TScrollbar", background=COLOR_PANEL_ALT, troughcolor=COLOR_PANEL)

    # ----------------------------------------------------------------- #
    # Layout
    # ----------------------------------------------------------------- #
    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bg=COLOR_PANEL, height=54)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        inner = tk.Frame(toolbar, bg=COLOR_PANEL)
        inner.pack(side=tk.LEFT, padx=10, pady=8)

        make_flat_button(inner, "Upload Image", self.upload_image).pack(side=tk.LEFT, padx=4)
        make_flat_button(inner, "Reset", self.reset_image, accent=COLOR_PANEL_ALT).pack(side=tk.LEFT, padx=4)
        make_flat_button(inner, "Save Image", self.save_image, accent=COLOR_ACCENT_2).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="No image loaded.")
        tk.Label(
            toolbar, textvariable=self.status_var, bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=14)

    def _build_body(self):
        body = tk.Frame(self.root, bg=COLOR_BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: image preview
        left_frame = tk.Frame(body, bg=COLOR_BG)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(left_frame, bg=COLOR_CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        hint = tk.Label(
            left_frame,
            text="Click anywhere on the image to pick a pixel color",
            bg=COLOR_BG,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 8, "italic"),
        )
        hint.pack(side=tk.BOTTOM, pady=(4, 0))

        # Right: scrollable controls
        right_container = tk.Frame(body, width=340, bg=COLOR_PANEL)
        right_container.pack(side=tk.RIGHT, fill=tk.Y)
        right_container.pack_propagate(False)

        control_canvas = tk.Canvas(right_container, bg=COLOR_PANEL, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_container, orient=tk.VERTICAL, command=control_canvas.yview)
        self.controls_frame = tk.Frame(control_canvas, bg=COLOR_PANEL, padx=14, pady=14)

        self.controls_frame.bind(
            "<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        control_canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            delta = -1 * (event.delta // 120) if event.delta else 0
            control_canvas.yview_scroll(int(delta), "units")

        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        control_canvas.bind_all("<Button-4>", lambda e: control_canvas.yview_scroll(-1, "units"))
        control_canvas.bind_all("<Button-5>", lambda e: control_canvas.yview_scroll(1, "units"))

        self._build_controls()

    def _build_controls(self):
        tk.Label(
            self.controls_frame,
            text="Pixel Color Tools",
            bg=COLOR_PANEL,
            fg=COLOR_ACCENT,
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        # ---- Color picker section ----
        self._section_label("Picked Color")
        swatch_row = tk.Frame(self.controls_frame, bg=COLOR_PANEL)
        swatch_row.pack(fill=tk.X, pady=(2, 4))

        self.picked_swatch = tk.Canvas(swatch_row, width=36, height=24, bg="#333333", highlightthickness=1,
                                        highlightbackground=COLOR_TEXT_MUTED)
        self.picked_swatch.pack(side=tk.LEFT, padx=(0, 8))

        self.picked_label = tk.Label(
            swatch_row, text="No color picked", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 9)
        )
        self.picked_label.pack(side=tk.LEFT)

        ttk.Separator(self.controls_frame, orient="horizontal").pack(fill=tk.X, pady=10)

        # ---- Color Replace ----
        self.color_replace_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.controls_frame,
            text="Enable Color Replace",
            variable=self.color_replace_var,
            command=self.update_image,
        ).pack(anchor="w", pady=2)

        replace_row = tk.Frame(self.controls_frame, bg=COLOR_PANEL)
        replace_row.pack(fill=tk.X, pady=(4, 4))

        self.replacement_swatch = tk.Canvas(
            replace_row, width=36, height=24, bg=bgr_to_hex(self.replacement_color_bgr),
            highlightthickness=1, highlightbackground=COLOR_TEXT_MUTED
        )
        self.replacement_swatch.pack(side=tk.LEFT, padx=(0, 8))

        make_flat_button(replace_row, "Choose Replacement", self.choose_replacement_color,
                          accent=COLOR_PANEL_ALT).pack(side=tk.LEFT)

        tk.Label(self.controls_frame, text="Tolerance", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9)).pack(
            anchor="w", pady=(8, 0)
        )
        self.tolerance_var = tk.IntVar(value=40)
        ttk.Scale(
            self.controls_frame, from_=0, to=150, orient=tk.HORIZONTAL, variable=self.tolerance_var,
            command=lambda v: self.update_image()
        ).pack(fill=tk.X)

        ttk.Separator(self.controls_frame, orient="horizontal").pack(fill=tk.X, pady=10)

        # ---- Posterize ----
        self.posterize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.controls_frame, text="Enable Posterize", variable=self.posterize_var, command=self.update_image
        ).pack(anchor="w", pady=2)

        tk.Label(self.controls_frame, text="Color Levels", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9)).pack(
            anchor="w", pady=(4, 0)
        )
        self.posterize_levels_var = tk.IntVar(value=6)
        ttk.Scale(
            self.controls_frame, from_=2, to=32, orient=tk.HORIZONTAL, variable=self.posterize_levels_var,
            command=lambda v: self.update_image()
        ).pack(fill=tk.X)

        ttk.Separator(self.controls_frame, orient="horizontal").pack(fill=tk.X, pady=10)

        # ---- Pixelate ----
        self.pixelate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.controls_frame, text="Enable Pixelate", variable=self.pixelate_var, command=self.update_image
        ).pack(anchor="w", pady=2)

        tk.Label(self.controls_frame, text="Block Size", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9)).pack(
            anchor="w", pady=(4, 0)
        )
        self.pixelate_size_var = tk.IntVar(value=8)
        ttk.Scale(
            self.controls_frame, from_=2, to=50, orient=tk.HORIZONTAL, variable=self.pixelate_size_var,
            command=lambda v: self.update_image()
        ).pack(fill=tk.X)

        ttk.Separator(self.controls_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        make_flat_button(
            self.controls_frame, "Reset All Options", self.reset_controls, accent=COLOR_PANEL_ALT
        ).pack(fill=tk.X, pady=4)

    def _section_label(self, text):
        tk.Label(
            self.controls_frame, text=text, bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 2))

    # ----------------------------------------------------------------- #
    # Image I/O
    # ----------------------------------------------------------------- #
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All files", "*.*")],
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the selected image file.")
            return

        self.original_image = img
        self.processed_image = img.copy()
        self.picked_color_bgr = None
        self.picked_label.config(text="No color picked")
        self.picked_swatch.configure(bg="#333333")
        self.status_var.set(f"Loaded: {path.split('/')[-1]}  ({img.shape[1]}x{img.shape[0]})")
        self.reset_controls(update=False)
        self.render_image(self.processed_image)

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("No Image", "There is no processed image to save.")
            return

        path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Bitmap", "*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        success = cv2.imwrite(path, self.processed_image)
        if success:
            messagebox.showinfo("Saved", f"Image saved successfully to:\n{path}")
        else:
            messagebox.showerror("Error", "Failed to save the image.")

    # ----------------------------------------------------------------- #
    # Color picking / replacement
    # ----------------------------------------------------------------- #
    def on_canvas_click(self, event):
        if self.original_image is None:
            return

        scale = self._render_scale
        off_x, off_y = self._render_offset
        rend_w, rend_h = self._render_size

        # Position relative to the rendered image (not the whole canvas)
        rel_x = event.x - off_x
        rel_y = event.y - off_y
        if rel_x < 0 or rel_y < 0 or rel_x >= rend_w or rel_y >= rend_h:
            return  # click was outside the image area

        img_x = int(rel_x / scale)
        img_y = int(rel_y / scale)

        base = self.processed_image if self.processed_image is not None else self.original_image
        h, w = base.shape[:2]
        img_x = min(max(img_x, 0), w - 1)
        img_y = min(max(img_y, 0), h - 1)

        bgr = base[img_y, img_x].tolist()
        self.picked_color_bgr = tuple(int(c) for c in bgr)
        hex_val = bgr_to_hex(self.picked_color_bgr)

        self.picked_swatch.configure(bg=hex_val)
        r, g, b = self.picked_color_bgr[2], self.picked_color_bgr[1], self.picked_color_bgr[0]
        self.picked_label.config(text=f"RGB({r},{g},{b})  {hex_val}")

        if self.color_replace_var.get():
            self.update_image()

    def choose_replacement_color(self):
        initial = bgr_to_hex(self.replacement_color_bgr)
        result = colorchooser.askcolor(color=initial, title="Choose replacement color")
        if result and result[1]:
            self.replacement_color_bgr = hex_to_bgr(result[1])
            self.replacement_swatch.configure(bg=result[1])
            self.update_image()

    # ----------------------------------------------------------------- #
    # Processing pipeline
    # ----------------------------------------------------------------- #
    def update_image(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()

        if self.pixelate_var.get():
            img = apply_pixelate(img, self.pixelate_size_var.get())

        if self.posterize_var.get():
            img = apply_posterize(img, self.posterize_levels_var.get())

        if self.color_replace_var.get() and self.picked_color_bgr is not None:
            img = apply_color_replace(
                img, self.picked_color_bgr, self.replacement_color_bgr, self.tolerance_var.get()
            )

        self.processed_image = img
        self.render_image(img)

    def reset_controls(self, update=True):
        self.color_replace_var.set(False)
        self.posterize_var.set(False)
        self.pixelate_var.set(False)
        self.tolerance_var.set(40)
        self.posterize_levels_var.set(6)
        self.pixelate_size_var.set(8)
        if update and self.original_image is not None:
            self.update_image()

    def reset_image(self):
        if self.original_image is None:
            return
        self.reset_controls(update=False)
        self.picked_color_bgr = None
        self.picked_label.config(text="No color picked")
        self.picked_swatch.configure(bg="#333333")
        self.processed_image = self.original_image.copy()
        self.render_image(self.processed_image)
        self.status_var.set("Reset to original image.")

    # ----------------------------------------------------------------- #
    # Rendering
    # ----------------------------------------------------------------- #
    def render_image(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        w, h = pil_img.size
        scale = min(self.DISPLAY_MAX_W / w, self.DISPLAY_MAX_H / h, 1.0)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self.display_photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width() or self.DISPLAY_MAX_W
        canvas_h = self.canvas.winfo_height() or self.DISPLAY_MAX_H
        x = max(canvas_w // 2, new_w // 2)
        y = max(canvas_h // 2, new_h // 2)
        self.canvas.create_image(x, y, image=self.display_photo, anchor="center")

        # Store mapping info so canvas clicks can be translated back to image pixels
        self._render_scale = scale
        self._render_offset = (x - new_w // 2, y - new_h // 2)
        self._render_size = (new_w, new_h)


def main():
    root = tk.Tk()
    app = PixelColorStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
