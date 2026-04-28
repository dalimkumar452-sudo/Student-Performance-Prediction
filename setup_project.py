import os

def create_project_structure():
    # প্রজেক্টের নাম
    project_name = "Student-Performance-Prediction"
    
    # যে ফোল্ডারগুলো তৈরি করতে হবে
    folders = [
        os.path.join(project_name, 'data'),
        os.path.join(project_name, 'notebooks'),
        os.path.join(project_name, 'src'),
        os.path.join(project_name, 'models'),
        os.path.join(project_name, 'outputs'),
        os.path.join(project_name, 'images'),
    ]
    
    # যে ফাইলগুলো তৈরি করতে হবে
    files = [
        os.path.join(project_name, 'src', 'preprocessing.py'),
        os.path.join(project_name, 'src', 'model_dev.py'),
        os.path.join(project_name, 'main.py'),
        os.path.join(project_name, 'requirements.txt'),
        os.path.join(project_name, 'README.md'),
        os.path.join(project_name, '.gitignore'),
    ]
    
    print(f"🚀 Creating structure for: {project_name}...")

    # ফোল্ডার তৈরি
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 Folder Created: {folder}")

    # ফাইল তৈরি (ফাঁকা ফাইল)
    for file in files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                if 'requirements.txt' in file:
                    f.write("pandas\nnumpy\nscikit-learn\nxgboost\nplotly\njoblib\nfastapi\nuvicorn")
                elif '.gitignore' in file:
                    f.write("data/\nmodels/*.pkl\nmodels/*.joblib\n__pycache__/\n.env\noutputs/*.html")
                else:
                    f.write(f"# {os.path.basename(file)} file")
            print(f"📄 File Created: {file}")

    print("\n✅ Project structure is ready! Now you can start coding.")

if __name__ == "__main__":
    create_project_structure()