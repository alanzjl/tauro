from sshkeyboard import listen_keyboard


def on_press(key):
    print(f"{key} pressed")


def on_release(key):
    print(f"{key} released")


listen_keyboard(on_press=on_press, on_release=on_release)  # Esc quits
