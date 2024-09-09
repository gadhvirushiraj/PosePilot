# import customtkinter as ctk
# from CTkMessagebox import CTkMessagebox
# from tkinter import *
# # import ttk
# import cv2
# from PIL import Image, ImageTk
# import os
# import get_skeleton as gs
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import numpy as np

# ctk.set_appearance_mode("System")
# ctk.set_default_color_theme("green")


# class GUI(ctk.CTk):
#     def __init__(self):
#         super().__init__()
#         self.title("PosePilot")
#         self.geometry("980x630")
#         self.resizable(False, False)
#         self.speed_factor = 1.0  # Default speed factor

#     def setup_ui(self):
#         self.setup_video_frame()
#         self.setup_skeleton_frame()
#         self.setup_graph_frame()
#         self.setup_control_frame()

#     def setup_video_frame(self):
#         self.video_frame = ctk.CTkFrame(
#             self, width=400, height=300)  # 1. Top left: Video
#         self.video_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
#         self.video_canvas = ctk.CTkCanvas(
#             self.video_frame, width=400, height=300)
#         self.video_canvas.pack()

#     def setup_skeleton_frame(self):
#         self.skeleton_frame = ctk.CTkFrame(
#             self, width=400, height=300)  # 2. Top right: Skeleton Pose
#         self.skeleton_frame.grid(
#             row=0, column=1, padx=5, pady=5, sticky="nsew")
#         self.skeleton_canvas = ctk.CTkCanvas(
#             self.skeleton_frame, width=400, height=300)
#         self.skeleton_canvas.pack()
#         # Add rendering code for skeleton frame here

#     def setup_graph_frame(self):
#         self.graph_frame = ctk.CTkFrame(
#             self, width=400, height=300)  # 3. Bottom: Graph
#         self.graph_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
#         self.graph_canvas = ctk.CTkCanvas(
#             self.graph_frame, width=400, height=300)
#         self.graph_canvas.pack()
#         # Add rendering code for graph frame here

#     def setup_control_frame(self):
#         # 4. Bottom right: Control Panel
#         self.control_frame = ctk.CTkFrame(self, width=400, height=300)
#         self.control_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

#         # Entry for video path
#         self.video_path_entry = ctk.CTkEntry(
#             self.control_frame, placeholder_text="Enter video path")
#         self.video_path_entry.grid(
#             row=0, column=0, padx=5, pady=5, sticky="ew")

#         # Button to load video
#         self.load_video_button = ctk.CTkButton(
#             self.control_frame, text="Load Video", command=self.load_video)
#         self.load_video_button.grid(
#             row=0, column=1, padx=5, pady=5, sticky="ew")

#         # Button to process video
#         self.process_video_button = ctk.CTkButton(
#             self.control_frame, text="Process Video", command=self.load_skeleton)
#         self.process_video_button.grid(
#             row=0, column=2, padx=5, pady=5, sticky="ew")

#         # separator
#         self.sep = ctk.CTkLabel(self.control_frame, text="", height=2, fg_color="gray")
#         self.sep.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        

#         # Model entry box
#         # self.model_entry = ctk.CTkEntry(
#         #     self.control_frame, placeholder_text="Enter model path")
#         # self.model_entry.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

#         # # Button to load model
#         # self.load_model_button = ctk.CTkButton(
#         #     self.control_frame, text="Load Model")
#         # self.load_model_button.grid(
#         #     row=2, column=1, padx=5, pady=5, sticky="ew")

#         # separator
#         self.sep2 = ctk.CTkLabel(self.control_frame, text="", height=2, fg_color="gray")
#         self.sep2.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

#         # Label for speed slider
#         self.speed_label = ctk.CTkLabel(
#             self.control_frame, text="Video speed [fast <-> slow]:")
#         self.speed_label.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

#         # Slider for adjusting video speed
#         self.speed_slider = ctk.CTkSlider(
#             self.control_frame, from_=1, to=10, variable=ctk.DoubleVar(), command=self.set_speed)
#         self.speed_slider.set(10)  # Default speed
#         self.speed_slider.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

#         # Button to replay video
#         self.replay_button = ctk.CTkButton(
#             self.control_frame, text="RUN", command=self.run)
#         self.replay_button.grid(row=5, columnspan=3,
#                                 padx=5, pady=5, sticky="ew")

#         # separator
#         self.sep3 = ctk.CTkLabel(self.control_frame, text="", height=2, fg_color="gray")
#         self.sep3.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

#         # graph control buttonsx
#         self.graph_contol_label = ctk.CTkLabel(
#             self.control_frame, text="Graph control:")
#         self.graph_contol_label.grid(
#             row=7, column=0, padx=5, pady=5, sticky="w")
#         self.graph_button_left = ctk.CTkButton(
#             self.control_frame, text="<< left")
#         self.graph_button_left.grid(
#             row=8, column=0, padx=5, pady=5, sticky="ew")
#         self.graph_button_right = ctk.CTkButton(
#             self.control_frame, text="right >>")
#         self.graph_button_right.grid(
#             row=8, column=1, padx=5, pady=5, sticky="ew")

#         # Configure column and row weights for control_frame
#         self.control_frame.grid_columnconfigure([0, 1, 2], weight=1)
#         self.control_frame.grid_rowconfigure(
#             [0, 1, 2, 3, 4, 5, 6, 7, 8], weight=1)

#     def load_video(self):
#         video_path = self.video_path_entry.get()

#         if not video_path:
#             CTkMessagebox(title="Warning Message!", message="Please provide video path",
#                           icon="warning", option_1="chillax bud!")
#             return

#         if not os.path.exists(video_path):
#             CTkMessagebox(title="Warning Message!", message="Provided video path doesn't exist",
#                           icon="warning", option_1="ya, i am dumb!")
#             return

#         if video_path:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print("Error: Unable to open video.")
#                 return

#             fps = 30  # Default value
#             self.video_canvas.config(width=400, height=300)
#             self.video_canvas.delete("all")

#             # Play video
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 resized_frame = cv2.resize(frame, (400, 300))
#                 resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
#                 img = Image.fromarray(resized_frame)
#                 img_tk = ImageTk.PhotoImage(image=img)

#                 self.video_canvas.create_image(0, 0, anchor="nw", image=img_tk)
#                 self.video_canvas.image = img_tk
#                 self.update_idletasks()
#                 self.update()
#                 delay = int(1000 / (fps))  # milliseconds
#                 self.after(delay)

#             cap.release()


#     def set_speed(self, speed):
#         self.speed_factor = float(speed)

#     def run(self):
#         video_path = self.video_path_entry.get()

#         # if not video_path or not model_path:
#         #     CTkMessagebox(title="Warning Message!", message="Please load the video and model beofre running", icon="warning", option_1="chill buddy!")
#         #     return

#         self.load_video()
#         self.play_skeleton()

#     def play_skeleton(self):
#         if not hasattr(self, 'skeleton_images') or not self.skeleton_images:
#             print("No skeleton images found")
#             return

#         self.skeleton_canvas.config(width=400, height=300)
#         self.skeleton_canvas.delete("all")

#         fps = 30  # Default value
#         print("Playing skeleton...")
#         print(len(self.skeleton_images))
#         for fig in self.skeleton_images:
#             # Convert the figure to an image
#             canvas = FigureCanvas(fig)
#             canvas.draw()

#             width, height = canvas.get_width_height()
#             s, (width, height) = canvas.print_to_buffer()
#             img = np.fromstring(s, np.uint8).reshape((height, width, 4))
#             img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

#             # Convert to Tkinter-compatible format
#             img = cv2.resize(img, (400, 300))
#             img = Image.fromarray(img)
#             img_tk = ImageTk.PhotoImage(image=img)

#             # Display the image on the canvas
#             self.skeleton_canvas.create_image(0, 0, anchor="nw", image=img_tk)
#             self.skeleton_canvas.image = img_tk

#             # Update the tkinter window
#             self.update_idletasks()
#             self.update()

#             # Set delay for the next frame
#             delay = int(1000 / fps)
#             self.after(delay)

#         print("Skeleton played!")


#     def load_skeleton(self):
#         print("Loading skeleton...")
#         video_path = self.video_path_entry.get()
#         label = "chair"
#         landmarks, landmark_mp_list = gs.give_landmarks(video_path, label, 30)

#         self.skeleton_images = []  # Reset skeleton images list
#         for data in landmark_mp_list:
#             frame = gs.make_skeleton_frame(data.pose_landmarks)
#             self.skeleton_images.append(frame)

#         print("Skeleton loaded!")


# if __name__ == "__main__":
#     gui = GUI()
#     gui.setup_ui()
#     gui.mainloop()
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import get_skeleton as gs
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import subprocess
import sys

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")


class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PosePilot")
        self.geometry("980x630")
        self.resizable(False, False)
        self.speed_factor = 1.0  # Default speed factor

    def setup_ui(self):
        self.setup_video_frame()
        self.setup_skeleton_frame()
        self.setup_graph_frame()
        self.setup_control_frame()

    def setup_video_frame(self):
        self.video_frame = ctk.CTkFrame(self, width=400, height=300)  # 1. Top left: Video
        self.video_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.video_canvas = ctk.CTkCanvas(self.video_frame, width=400, height=300)
        self.video_canvas.pack()

    def setup_skeleton_frame(self):
        self.skeleton_frame = ctk.CTkFrame(self, width=400, height=300)  # 2. Top right: Skeleton Pose
        self.skeleton_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.skeleton_canvas = ctk.CTkCanvas(self.skeleton_frame, width=400, height=300)
        self.skeleton_canvas.pack()

    def setup_graph_frame(self):
        self.graph_frame = ctk.CTkFrame(self, width=400, height=300)  # 3. Bottom: Graph
        self.graph_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.graph_canvas = ctk.CTkCanvas(self.graph_frame, width=400, height=300)
        self.graph_canvas.pack()

    def setup_control_frame(self):
        self.control_frame = ctk.CTkFrame(self, width=400, height=300)  # 4. Bottom right: Control Panel
        self.control_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.video_path_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Enter video path")
        self.video_path_entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.load_video_button = ctk.CTkButton(self.control_frame, text="Load Video", command=self.load_video)
        self.load_video_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.process_video_button = ctk.CTkButton(self.control_frame, text="Process Video", command=self.run_process)
        self.process_video_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.speed_label = ctk.CTkLabel(self.control_frame, text="Video speed [fast <-> slow]:")
        self.speed_label.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        self.speed_slider = ctk.CTkSlider(self.control_frame, from_=1, to=10, variable=ctk.DoubleVar(), command=self.set_speed)
        self.speed_slider.set(10)
        self.speed_slider.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        self.replay_button = ctk.CTkButton(self.control_frame, text="RUN", command=self.run)
        self.replay_button.grid(row=5, columnspan=3, padx=5, pady=5, sticky="ew")

        self.control_frame.grid_columnconfigure([0, 1, 2], weight=1)
        self.control_frame.grid_rowconfigure([0, 1, 2, 3, 4, 5, 6, 7, 8], weight=1)

    def load_video(self):
        video_path = self.video_path_entry.get()
        if not video_path:
            CTkMessagebox(title="Warning!", message="Please provide a video path", icon="warning", option_1="OK")
            return
        if not os.path.exists(video_path):
            CTkMessagebox(title="Warning!", message="The video path doesn't exist", icon="warning", option_1="OK")
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return

        self.video_canvas.config(width=400, height=300)
        self.video_canvas.delete("all")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (400, 300))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor="nw", image=img_tk)
            self.video_canvas.image = img_tk
            self.update_idletasks()
            self.update()
            delay = int(1000 / 30)
            self.after(delay)
        cap.release()

    def run_process(self):
    # Use the current Python executable to run the script
        subprocess.run([sys.executable, "main.py", "false_1.mp4"])
        self.load_graph()

    def load_graph(self):
        graph_path = "correction_output.png"
        if not os.path.exists(graph_path):
            CTkMessagebox(title="Error!", message="Graph file not found", icon="error", option_1="OK")
            return
        graph_image = Image.open(graph_path)
        graph_image = graph_image.resize((400, 300), Image.LANCZOS)
        graph_tk = ImageTk.PhotoImage(graph_image)
        self.graph_canvas.create_image(0, 0, anchor="nw", image=graph_tk)
        self.graph_canvas.image = graph_tk

    def set_speed(self, speed):
        self.speed_factor = float(speed)

    def run(self):
        self.load_video()
        self.play_skeleton()

    def play_skeleton(self):
        if not hasattr(self, 'skeleton_images') or not self.skeleton_images:
            print("No skeleton images found")
            return
        self.skeleton_canvas.config(width=400, height=300)
        self.skeleton_canvas.delete("all")
        fps = 30
        for fig in self.skeleton_images:
            canvas = FigureCanvas(fig)
            canvas.draw()
            width, height = canvas.get_width_height()
            s, (width, height) = canvas.print_to_buffer()
            img = np.fromstring(s, np.uint8).reshape((height, width, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            img = cv2.resize(img, (400, 300))
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.skeleton_canvas.create_image(0, 0, anchor="nw", image=img_tk)
            self.skeleton_canvas.image = img_tk
            self.update_idletasks()
            self.update()
            delay = int(1000 / fps)
            self.after(delay)

    def load_skeleton(self):
        print("Loading skeleton...")
        video_path = self.video_path_entry.get()
        label = "chair"
        landmarks, landmark_mp_list = gs.give_landmarks(video_path, label, 30)
        self.skeleton_images = []
        for data in landmark_mp_list:
            frame = gs.make_skeleton_frame(data.pose_landmarks)
            self.skeleton_images.append(frame)
        print("Skeleton loaded!")


if __name__ == "__main__":
    gui = GUI()
    gui.setup_ui()
    gui.mainloop()
