from core.inspector import Inspector
from gui.main_window import MainWindow

def main():
    inspector = Inspector()
    window = MainWindow(inspector)
    try:
        window.run()
    finally:
        inspector.shutdown()



if __name__ == "__main__":
    main()