def save_to_csv(data, path='', filename):
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        data.to_csv(full_path, index=True)
        print(f"File saved as: {full_path}")
    else:
        print(f"File {full_path} already exists.")
