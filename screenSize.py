import ctypes
import pyautogui
import tkinter as tk

root = tk.Tk()

width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("TK Size", "width",width," height",height)

ctypes.windll.user32.SetProcessDPIAware()

screenSize = pyautogui.size()
print(screenSize)
# (width=1920, height=1080)