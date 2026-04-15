from core.inspector import Inspector

def main():
    inspector = Inspector()
    try:
        # inspector.manual_control()
        inspector.scan_plate()
    finally:
        inspector.shutdown()



if __name__ == "__main__":
    main()