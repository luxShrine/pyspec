from pyspectral.config import DATA_DIR, RAW_DATA_DIR, READY_DATA_DIR
import pyspectral.data.io as pi


def do_conversion():
    csv = None
    for f in DATA_DIR.glob("*.csv"):
        user_input = input(f"Use '{f}' as metadata container? (y/n) ")
        user_input = user_input.lower()
        if "y" in user_input or "yes" in user_input:
            csv = f
            break
        elif "n" in user_input or "no" in user_input:
            continue
        print(f"Unknown input recieved: {user_input}")

    if csv is None:
        print("Could not find suitable metadata file, exiting.")
        return

    print("Converting files...")
    pi.convert_raw_class(csv=csv, base=DATA_DIR)


def main():
    while True:
        user_input = input("Convert raw HSI txt files to formatted files? (y/n) ")
        user_input = user_input.lower()
        if "y" in user_input or "yes" in user_input:
            do_conversion()
            print("Done.")
            break
        elif "n" in user_input or "no" in user_input:
            print("Exiting...")
            break

        print(f"Unknown response recieved {user_input}")
        continue


if __name__ == "__main__":
    main()
