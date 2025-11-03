import pandas as pd
import os

def main():
    file_name = input("Nome do arquivo .parquet (sem .parquet): ").strip() + ".parquet"
    txt_name = "input.txt"

    # Load existing dataset if available
    if os.path.exists(file_name):
        df = pd.read_parquet(file_name)
        print(f"Arquivo existente: {file_name} ({len(df)} linhas)")
    else:
        df = pd.DataFrame(columns=["text", "label"])
        print(f"Arquivo parquet criado: {file_name}")

    # Ensure text file exists and is cleared
    open(txt_name, "w", encoding="utf-8").close()
    print(f"Cole o texto em '{txt_name}', e pressione Enter.")

    while True:
        input("Pressione Enter para salvar.")

        # Read text content
        with open(txt_name, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"{txt_name} vazio!")
            continue

        # Append text to dataframe
        new_row = pd.DataFrame({"text": [text], "label": ["human"]})
        df = pd.concat([df, new_row], ignore_index=True)

        # Save to parquet
        df.to_parquet(file_name, index=False)
        print(f"Texto adicionado em '{file_name}' ({len(df)} linhas)")

        # Clear text file
        open(txt_name, "w", encoding="utf-8").close()
        print(f"Conteudo de '{txt_name}' deletado.")

        # Ask to continue
        cont = input("Adicionar outro? (y/n): ").strip().lower()
        if cont != "y":
            break


if __name__ == "__main__":
    main()
