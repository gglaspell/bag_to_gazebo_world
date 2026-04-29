#!/usr/bin/env python3
"""
bag_to_gazebo_gui.py
Tkinter GUI launcher for the bag-to-gazebo-world Docker tool.
"""

import subprocess
import threading
import shlex
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (dark theme)
# ──────────────────────────────────────────────────────────────────────────────
BG          = "#1e1e2e"
SURFACE     = "#2a2a3e"
SURFACE2    = "#313150"
ACCENT      = "#7c9ef5"
ACCENT_HOV  = "#9db5ff"
TEXT        = "#cdd6f4"
TEXT_MUTED  = "#7f849c"
TEXT_INV    = "#1e1e2e"
BORDER      = "#45475a"
SUCCESS     = "#a6e3a1"
ERROR       = "#f38ba8"
WARNING     = "#fab387"
YELLOW      = "#f9e2af"
ENTRY_BG    = "#181825"
RUN_BG      = "#a6e3a1"
RUN_FG      = "#1e1e2e"
STOP_BG     = "#f38ba8"
STOP_FG     = "#1e1e2e"

GAZEBO_MATERIALS = [
    "Gazebo/Grey",
    "Gazebo/White",
    "Gazebo/DarkGrey",
    "Gazebo/Bricks",
    "Gazebo/Wood",
    "Gazebo/WoodFloor",
    "Gazebo/CeilingTiled",
    "Gazebo/Grass",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helper widgets
# ──────────────────────────────────────────────────────────────────────────────

def make_label(parent, text, muted=False, **kw):
    fg = TEXT_MUTED if muted else TEXT
    bg = parent["bg"] if hasattr(parent, "__getitem__") else BG
    return tk.Label(parent, text=text, bg=bg, fg=fg,
                    font=("Segoe UI", 9), **kw)

def make_entry(parent, textvariable, width=22, **kw):
    return tk.Entry(parent, textvariable=textvariable, width=width,
                    bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                    relief="flat", highlightthickness=1,
                    highlightbackground=BORDER, highlightcolor=ACCENT,
                    font=("Consolas", 9), **kw)

def make_section(parent, title):
    return tk.LabelFrame(parent, text="  " + title + "  ",
                         bg=SURFACE, fg=ACCENT,
                         font=("Segoe UI", 9, "bold"),
                         relief="flat", bd=1,
                         highlightthickness=1,
                         highlightbackground=BORDER,
                         padx=10, pady=8)

def browse_dir(var):
    path = filedialog.askdirectory()
    if path:
        var.set(path)

def styled_button(parent, text, command, bg=ACCENT, fg=TEXT_INV, font_size=9):
    return tk.Button(parent, text=text, command=command,
                     bg=bg, fg=fg, activebackground=ACCENT_HOV,
                     activeforeground=TEXT_INV, relief="flat",
                     font=("Segoe UI", font_size, "bold"),
                     padx=10, pady=4, cursor="hand2", bd=0)

# ──────────────────────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────────────────────

class BagToGazeboGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("bag-to-gazebo-world  \u00b7  Docker Launcher")
        self.geometry("1280x820")
        self.minsize(1000, 600)
        self.configure(bg=BG)
        self._process = None
        self._running = False
        self._tip_win = None
        self._build_vars()
        self._build_ui()
        self._update_command()

    # ── variables ─────────────────────────────────────────────────────────────
    def _build_vars(self):
        self.v_image        = tk.StringVar(value="bag-to-gazebo-world")
        self.v_bagpath      = tk.StringVar()
        self.v_outputdir    = tk.StringVar()
        self.v_model_name   = tk.StringVar(value="bag_environment")
        self.v_gz_material  = tk.StringVar(value="Gazebo/Grey")
        self.v_pc_topic     = tk.StringVar(value="points")
        self.v_odom_topic   = tk.StringVar(value="")
        self.v_voxel_size   = tk.StringVar(value="0.05")
        self.v_icp_dist     = tk.StringVar(value="0.2")
        self.v_icp_fit      = tk.StringVar(value="0.6")
        self.v_odom_latency = tk.StringVar(value="0.5")
        self.v_poisson      = tk.StringVar(value="9")
        self.v_min_density  = tk.StringVar(value="1.0")
        self.v_max_vtx_dist = tk.StringVar(value="0.15")
        self.v_decimate     = tk.StringVar(value="")
        self.v_level_floor  = tk.BooleanVar(value=False)
        self.v_workers      = tk.StringVar(value="4")
        self.v_loop_closure = tk.BooleanVar(value=False)
        self.v_lc_radius    = tk.StringVar(value="10.0")
        self.v_lc_fitness   = tk.StringVar(value="0.3")
        self.v_lc_interval  = tk.StringVar(value="10")

        for attr in vars(self):
            v = getattr(self, attr)
            if isinstance(v, (tk.StringVar, tk.BooleanVar)):
                v.trace_add("write", lambda *_: self._update_command())

    # ── UI layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=SURFACE2, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="\U0001f6f0  bag-to-gazebo-world",
                 bg=SURFACE2, fg=ACCENT,
                 font=("Segoe UI", 15, "bold")).pack(side="left", padx=18)
        tk.Label(hdr, text="Docker GUI Launcher",
                 bg=SURFACE2, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side="left", padx=4)

        # Horizontal paned window: params LEFT | output RIGHT
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG,
                               sashwidth=6, sashrelief="flat", sashpad=2)
        paned.pack(fill="both", expand=True)

        left_frame  = tk.Frame(paned, bg=BG)
        right_frame = tk.Frame(paned, bg=BG)
        paned.add(left_frame,  minsize=420)
        paned.add(right_frame, minsize=360)

        # ── Left: scrollable params ────────────────────────────────────────
        canvas = tk.Canvas(left_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical",
                                   command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=BG)
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        # Bind mousewheel only when cursor is over the left panel
        canvas.bind("<Enter>",
            lambda e: canvas.bind_all("<MouseWheel>",
                lambda ev: canvas.yview_scroll(
                    int(-1 * (ev.delta / 120)), "units")))
        canvas.bind("<Leave>",
            lambda e: canvas.unbind_all("<MouseWheel>"))

        self._build_params(self.scroll_frame)

        # ── Right: command preview + controls + output log ─────────────────
        self._build_output_area(right_frame)

    # ── Parameter sections ────────────────────────────────────────────────────
    def _build_params(self, parent):
        pad = {"padx": 14, "pady": 6, "fill": "x"}

        docker_sec = make_section(parent, "Docker")
        docker_sec.pack(**pad)
        self._row(docker_sec, "Image name", self.v_image,
                  tip="Docker image tag to run")

        req_sec = make_section(parent, "Required Paths")
        req_sec.pack(**pad)
        self._path_row(req_sec, "Bag folder  (bagpath) *", self.v_bagpath,
                       tip="Path to the ROS 2 bag folder on the host")
        self._path_row(req_sec, "Output directory *", self.v_outputdir,
                       tip="Directory where Gazebo files will be written")

        basic_sec = make_section(parent, "Basic Settings")
        basic_sec.pack(**pad)
        self._row(basic_sec, "--model_name", self.v_model_name,
                  tip="Name of the generated Gazebo model  [default: bag_environment]")
        self._combo_row(basic_sec, "--gazebo_material", self.v_gz_material,
                        GAZEBO_MATERIALS,
                        tip="Surface material applied in Gazebo")
        self._row(basic_sec, "--pc_topic", self.v_pc_topic,
                  tip="ROS 2 PointCloud2 topic name  [default: points]")
        self._row(basic_sec, "--odom_topic", self.v_odom_topic,
                  tip="Optional odometry topic for better frame registration  [default: none]")

        reg_sec = make_section(parent, "Registration")
        reg_sec.pack(**pad)
        self._row(reg_sec, "--voxel_size", self.v_voxel_size,
                  tip="Voxel downsampling size in metres  [default: 0.05]")
        self._row(reg_sec, "--icp_dist_thresh", self.v_icp_dist,
                  tip="ICP max correspondence distance in metres  [default: 0.2]")
        self._row(reg_sec, "--icp_fitness_thresh", self.v_icp_fit,
                  tip="Minimum ICP fitness score to accept a frame (0.0-1.0)  [default: 0.6]")
        self._row(reg_sec, "--odom_max_latency", self.v_odom_latency,
                  tip="Max odometry timestamp age (s) before falling back to identity  [default: 0.5]")

        mesh_sec = make_section(parent, "Mesh Reconstruction")
        mesh_sec.pack(**pad)
        self._row(mesh_sec, "--poisson_depth", self.v_poisson,
                  tip="Poisson reconstruction depth (8-11 recommended)  [default: 9]")
        self._row(mesh_sec, "--min_density_percentile", self.v_min_density,
                  tip="Removes the lowest density % of the mesh after reconstruction  [default: 1.0]")
        self._row(mesh_sec, "--max_vertex_distance", self.v_max_vtx_dist,
                  tip="Removes mesh vertices farther than this from any input point (m)  [default: 0.15]")
        self._row(mesh_sec, "--decimate_target", self.v_decimate,
                  tip="Triangle reduction: <=1.0 = ratio (e.g. 0.25), >1 = absolute count  [default: none]")

        post_sec = make_section(parent, "Post-Processing")
        post_sec.pack(**pad)
        self._check_row(post_sec, "--level_floor", self.v_level_floor,
                        tip="Automatically detect and align the floor plane to Z=0")
        self._row(post_sec, "--workers", self.v_workers,
                  tip="Number of parallel CPU workers for KDTree queries  [default: 4]")

        lc_sec = make_section(parent, "Loop Closure")
        lc_sec.pack(**pad)
        self._check_row(lc_sec, "--enable_loop_closure", self.v_loop_closure,
                        tip="Enable loop closure detection for longer trajectories")
        self._row(lc_sec, "--loop_closure_radius", self.v_lc_radius,
                  tip="Search radius for loop closures in metres  [default: 10.0]")
        self._row(lc_sec, "--loop_closure_fitness_thresh", self.v_lc_fitness,
                  tip="Minimum ICP fitness for a loop closure to be accepted  [default: 0.3]")
        self._row(lc_sec, "--loop_closure_search_interval", self.v_lc_interval,
                  tip="Check for loop closures every N frames  [default: 10]")

        tk.Frame(parent, bg=BG, height=12).pack()

    # ── Row helpers ───────────────────────────────────────────────────────────
    def _row(self, parent, label, var, width=28, tip=""):
        f = tk.Frame(parent, bg=parent["bg"])
        f.pack(fill="x", pady=3)
        lbl = make_label(f, label)
        lbl.pack(side="left", padx=(0, 8))
        if tip:
            lbl.bind("<Enter>", lambda e, t=tip: self._show_tip(e, t))
            lbl.bind("<Leave>", lambda e: self._hide_tip())
        make_entry(f, var, width=width).pack(side="left", padx=2)
        make_label(f, "i", muted=True).pack(side="left", padx=(4, 0))

    def _path_row(self, parent, label, var, tip=""):
        f = tk.Frame(parent, bg=parent["bg"])
        f.pack(fill="x", pady=3)
        make_label(f, label).pack(side="left", padx=(0, 8))
        make_entry(f, var, width=36).pack(side="left", padx=2)
        styled_button(f, "Browse...", lambda v=var: browse_dir(v),
                      bg=SURFACE2, fg=ACCENT).pack(side="left", padx=(4, 0))

    def _combo_row(self, parent, label, var, values, tip=""):
        f = tk.Frame(parent, bg=parent["bg"])
        f.pack(fill="x", pady=3)
        make_label(f, label).pack(side="left", padx=(0, 8))
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TCombobox",
                        fieldbackground=ENTRY_BG, background=ENTRY_BG,
                        foreground=TEXT, selectbackground=ACCENT,
                        arrowcolor=ACCENT, bordercolor=BORDER)
        ttk.Combobox(f, textvariable=var, values=values,
                     state="readonly", width=24,
                     style="Dark.TCombobox").pack(side="left", padx=2)

    def _check_row(self, parent, label, var, tip=""):
        f = tk.Frame(parent, bg=parent["bg"])
        f.pack(fill="x", pady=3)
        tk.Checkbutton(f, text=label, variable=var,
                       bg=parent["bg"], fg=TEXT,
                       activebackground=parent["bg"], activeforeground=ACCENT,
                       selectcolor=ENTRY_BG, font=("Segoe UI", 9),
                       cursor="hand2").pack(side="left")
        if tip:
            make_label(f, "   " + tip, muted=True).pack(side="left")

    # ── Tooltip ───────────────────────────────────────────────────────────────
    def _show_tip(self, event, text):
        self._hide_tip()
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + 20
        self._tip_win = tk.Toplevel(self)
        self._tip_win.wm_overrideredirect(True)
        self._tip_win.wm_geometry("+" + str(x) + "+" + str(y))
        tk.Label(self._tip_win, text=text,
                 bg="#f9e2af", fg="#1e1e2e",
                 font=("Segoe UI", 8), padx=6, pady=3,
                 relief="solid", bd=1).pack()

    def _hide_tip(self):
        if self._tip_win:
            self._tip_win.destroy()
            self._tip_win = None

    # ── Right panel: command preview + controls + output ──────────────────────
    def _build_output_area(self, parent):
        # Command preview header
        cmd_hdr = tk.Frame(parent, bg=SURFACE, pady=6)
        cmd_hdr.pack(fill="x")
        tk.Label(cmd_hdr, text="Generated Command",
                 bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=14)
        styled_button(cmd_hdr, "Copy", self._copy_command,
                      bg=SURFACE2, fg=ACCENT).pack(side="right", padx=10)

        self.cmd_text = tk.Text(parent, height=5,
                                bg=ENTRY_BG, fg=YELLOW,
                                font=("Consolas", 8), relief="flat",
                                state="disabled", wrap="word",
                                insertbackground=TEXT,
                                highlightthickness=1,
                                highlightbackground=BORDER)
        self.cmd_text.pack(fill="x", padx=0)

        # Run / Stop controls
        ctrl = tk.Frame(parent, bg=SURFACE2, pady=8)
        ctrl.pack(fill="x")
        self.run_btn = styled_button(ctrl, "\u25b6  Run", self._run,
                                     bg=RUN_BG, fg=RUN_FG, font_size=10)
        self.run_btn.pack(side="left", padx=14)
        self.stop_btn = styled_button(ctrl, "\u25a0  Stop", self._stop,
                                      bg=STOP_BG, fg=STOP_FG, font_size=10)
        self.stop_btn.pack(side="left", padx=4)
        self.stop_btn.config(state="disabled")
        styled_button(ctrl, "Clear", self._clear_output,
                      bg=SURFACE, fg=TEXT_MUTED).pack(side="left", padx=12)
        self.status_label = tk.Label(ctrl, text="\u25cf Idle",
                                     bg=SURFACE2, fg=TEXT_MUTED,
                                     font=("Segoe UI", 9))
        self.status_label.pack(side="right", padx=14)

        # Output log (expands to fill remaining height)
        log_hdr = tk.Frame(parent, bg=BG, pady=4)
        log_hdr.pack(fill="x")
        tk.Label(log_hdr, text="Output Log",
                 bg=BG, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=14)

        self.output = scrolledtext.ScrolledText(
            parent, bg=ENTRY_BG, fg=TEXT,
            font=("Consolas", 8), relief="flat",
            state="disabled", wrap="word",
            highlightthickness=1, highlightbackground=BORDER)
        self.output.pack(fill="both", expand=True)
        self.output.tag_config("stderr", foreground=ERROR)
        self.output.tag_config("info",   foreground=SUCCESS)
        self.output.tag_config("cmd",    foreground=YELLOW)

    # ── Command builder ───────────────────────────────────────────────────────
    def _build_command(self):
        bagpath   = self.v_bagpath.get().strip()
        outputdir = self.v_outputdir.get().strip()
        image     = self.v_image.get().strip() or "bag-to-gazebo-world"

        if not bagpath or not outputdir:
            return None, "bagpath and outputdir are required."

        bag_host = os.path.abspath(bagpath)
        out_host = os.path.abspath(outputdir)

        cmd = [
            "docker", "run", "--rm",
            "-v", bag_host + ":/app/bag_input:ro",
            "-v", out_host + ":/app/output",
            image,
            "/app/bag_input",
            "/app/output",
        ]

        def add(flag, var, default=""):
            v = var.get().strip()
            if v and v != default:
                cmd.extend([flag, v])

        add("--model_name",      self.v_model_name,  "bag_environment")
        add("--gazebo_material", self.v_gz_material, "Gazebo/Grey")
        add("--pc_topic",        self.v_pc_topic,    "points")

        if self.v_odom_topic.get().strip():
            cmd.extend(["--odom_topic", self.v_odom_topic.get().strip()])

        add("--voxel_size",             self.v_voxel_size,   "0.05")
        add("--icp_dist_thresh",        self.v_icp_dist,     "0.2")
        add("--icp_fitness_thresh",     self.v_icp_fit,      "0.6")
        add("--odom_max_latency",       self.v_odom_latency, "0.5")
        add("--poisson_depth",          self.v_poisson,      "9")
        add("--min_density_percentile", self.v_min_density,  "1.0")
        add("--max_vertex_distance",    self.v_max_vtx_dist, "0.15")
        add("--workers",                self.v_workers,      "4")

        if self.v_decimate.get().strip():
            cmd.extend(["--decimate_target", self.v_decimate.get().strip()])

        if self.v_level_floor.get():
            cmd.append("--level_floor")

        if self.v_loop_closure.get():
            cmd.append("--enable_loop_closure")
            add("--loop_closure_radius",          self.v_lc_radius,   "10.0")
            add("--loop_closure_fitness_thresh",  self.v_lc_fitness,  "0.3")
            add("--loop_closure_search_interval", self.v_lc_interval, "10")

        return cmd, None

    def _update_command(self):
        cmd, err = self._build_command()
        self.cmd_text.config(state="normal")
        self.cmd_text.delete("1.0", "end")
        if err:
            self.cmd_text.insert("end", "# " + err)
        else:
            pretty = " \\\n  ".join(shlex.quote(a) for a in cmd)
            self.cmd_text.insert("end", pretty)
        self.cmd_text.config(state="disabled")

    def _copy_command(self):
        cmd, _ = self._build_command()
        if cmd:
            text = " \\\n  ".join(shlex.quote(a) for a in cmd)
            self.clipboard_clear()
            self.clipboard_append(text)
            self._log("Command copied to clipboard.\n", "info")

    # ── Run / Stop ─────────────────────────────────────────────────────────────
    def _run(self):
        if self._running:
            return
        cmd, err = self._build_command()
        if err:
            messagebox.showerror("Missing input", err)
            return

        bagpath   = self.v_bagpath.get().strip()
        outputdir = self.v_outputdir.get().strip()

        if not os.path.isdir(bagpath):
            messagebox.showerror("Path error",
                                 "Bag folder not found:\n" + bagpath)
            return

        os.makedirs(outputdir, exist_ok=True)

        self._running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="\u25cf Running...", fg=SUCCESS)

        pretty = " \\\n  ".join(shlex.quote(a) for a in cmd)
        self._log("$ " + pretty + "\n\n", "cmd")

        threading.Thread(target=self._run_thread,
                         args=(cmd,), daemon=True).start()

    def _run_thread(self, cmd):
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self._process.stdout:
                self._log(line)
            self._process.wait()
            rc = self._process.returncode
            if rc == 0:
                self._log("\n\u2714 Finished successfully.\n", "info")
                self.after(0, lambda: self.status_label.config(
                    text="\u25cf Done", fg=SUCCESS))
            else:
                self._log("\n\u2716 Process exited with code "
                          + str(rc) + ".\n", "stderr")
                self.after(0, lambda: self.status_label.config(
                    text="\u25cf Error (exit " + str(rc) + ")", fg=ERROR))
        except FileNotFoundError:
            self._log(
                "\u2716 docker not found. "
                "Is Docker installed and in your PATH?\n", "stderr")
            self.after(0, lambda: self.status_label.config(
                text="\u25cf Error", fg=ERROR))
        except Exception as exc:
            self._log("\u2716 " + str(exc) + "\n", "stderr")
            self.after(0, lambda: self.status_label.config(
                text="\u25cf Error", fg=ERROR))
        finally:
            self._running = False
            self._process = None
            self.after(0, lambda: self.run_btn.config(state="normal"))
            self.after(0, lambda: self.stop_btn.config(state="disabled"))

    def _stop(self):
        if self._process:
            self._process.terminate()
            self._log("\n\u23f9 Process terminated by user.\n", "stderr")
        self._running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="\u25cf Stopped", fg=WARNING)

    def _log(self, text, tag=None):
        self.output.config(state="normal")
        if tag:
            self.output.insert("end", text, tag)
        else:
            self.output.insert("end", text)
        self.output.see("end")
        self.output.config(state="disabled")

    def _clear_output(self):
        self.output.config(state="normal")
        self.output.delete("1.0", "end")
        self.output.config(state="disabled")
        self.status_label.config(text="\u25cf Idle", fg=TEXT_MUTED)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = BagToGazeboGUI()
    app.mainloop()
