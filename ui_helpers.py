import tkinter as tk


def tkinter_setup(app):
    root = tk.Tk()
    root.geometry("800x600")

    from PIL import Image, ImageTk
    bg_img = ImageTk.PhotoImage(Image.open("visuals/menu_background.jpg").resize((800, 600)))

    canvas = tk.Canvas(root, width=800, height=600, highlightthickness=0)
    canvas.place(x=0, y=0, relwidth=1, relheight=1)
    canvas.create_image(0, 0, anchor="nw", image=bg_img)
    canvas.image = bg_img

    # Buttons
    cup_1_button = make_button(
        root=root,
        command=lambda *args: app.pour_cup_button(root, [app.pour_cup_1]),
        text="Water",
        width=25,
        height=1,
        wraplength=0
    )
    cup_2_button = make_button(
        root=root,
        command=lambda *args: app.pour_cup_button(root, [app.pour_cup_2]),
        text="Water",
        width=25,
        height=1,
        wraplength=0
    )
    mix_button = make_button(
        root=root,
        command=lambda *args: app.pour_cup_button(root, [app.pour_cup_1, app.pour_cup_2]),
        text="Mixed (so still just water)",
        width=25,
        height=1,
        wraplength=0
    )

    canvas.create_text(400, 295, text="Pour Cup 1", font=("Arial", 8), fill="gray")
    canvas.create_text(400, 355, text="Pour Cup 2", font=("Arial", 8), fill="gray")

    cup_1_button.place(relx=0.5, rely=0.45, anchor="center")
    cup_2_button.place(relx=0.5, rely=0.55, anchor="center")
    mix_button.place(relx=0.5,   rely=0.65, anchor="center")

    root.mainloop()


def make_button(root=None,
                text="Click Me",
                command=None,
                activebackground="blue",
                activeforeground="white",
                anchor="center",
                bd=3,
                bg="lightgray",
                cursor="hand2",
                disabledforeground="gray",
                fg="black",
                font=("Arial", 12),
                height=2,
                highlightbackground="black",
                highlightcolor="green",
                highlightthickness=2,
                justify="center",
                overrelief="raised",
                padx=10,
                pady=5,
                width=15,
                wraplength=100):
    button = tk.Button(root,
                       text=text,
                       command=lambda: command(root),
                       activebackground=activebackground,
                       activeforeground=activeforeground,
                       anchor=anchor,
                       bd=bd,
                       bg=bg,
                       cursor=cursor,
                       disabledforeground=disabledforeground,
                       fg=fg,
                       font=font,
                       height=height,
                       highlightbackground=highlightbackground,
                       highlightcolor=highlightcolor,
                       highlightthickness=highlightthickness,
                       justify=justify,
                       overrelief=overrelief,
                       padx=padx,
                       pady=pady,
                       width=width,
                       wraplength=wraplength)
    return button
