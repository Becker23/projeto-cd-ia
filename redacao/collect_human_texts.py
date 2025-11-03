import pandas as pd
import os

def main():
    file_name = input("Enter Parquet filename (without .parquet): ").strip() + ".parquet"
    txt_name = "input.txt"

    # Load existing dataset if available
    if os.path.exists(file_name):
        df = pd.read_parquet(file_name)
        print(f"📂 Loaded existing file: {file_name} ({len(df)} entries)")
    else:
        df = pd.DataFrame(columns=["text", "label"])
        print(f"✅ Created new Parquet file: {file_name}")

    # Ensure text file exists and is cleared
    open(txt_name, "w", encoding="utf-8").close()
    print(f"📝 Paste your text into '{txt_name}', then press Enter here.")

    while True:
        input("➡️  When you're done pasting the text, press Enter to save it...")

        # Read text content
        with open(txt_name, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print("⚠️  The text file is empty. Please paste something and try again.")
            continue

        # Append text to dataframe
        new_row = pd.DataFrame({"text": [text], "label": ["human"]})
        df = pd.concat([df, new_row], ignore_index=True)

        # Save to parquet
        df.to_parquet(file_name, index=False)
        print(f"✅ Text added to '{file_name}' ({len(df)} total entries)")

        # Clear text file
        open(txt_name, "w", encoding="utf-8").close()
        print(f"🧹 '{txt_name}' cleared. Ready for next text.")

        # Ask to continue
        cont = input("Add another? (y/n): ").strip().lower()
        if cont != "y":
            print("👋 Done! Exiting.")
            break


if __name__ == "__main__":
    main()
