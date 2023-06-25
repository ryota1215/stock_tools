from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import requests


def send_message(text):
    headers = {
        "Authorization": "Bearer y0s3vexjvC0Nzy3uEtRPv07d2PZsNHZw0qZ6l6SIRWE",
    }
    files = {
        "message": (None, text),
    }
    requests.post("https://notify-api.line.me/api/notify", headers=headers, files=files)


def g_auth():
    try:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
    except Exception as e:
        send_message(f"GCP_Error\n {e}")
        assert False, f"GCP_Error\n {e}"

    return drive


def g_uploder(drive, path, g_filename):
    """
    param : drive GoogleDrive(gauth)
    param : path 保存するファイルのパス
    param : g_filename g-drive保存先のフォルダ名
    """
    folder_id = drive.ListFile({"q": f'title = "{g_filename}"'}).GetList()[0]["id"]
    f = drive.CreateFile({"parents": [{"id": folder_id}]})
    f.SetContentFile(path)
    f["title"] = os.path.basename(path)

    existing_files = drive.ListFile(
        {"q": f"'{folder_id}' in parents and title = '{os.path.basename(path)}'"}
    ).GetList()

    if len(existing_files) != 0:
        # File already exists, update it
        id = existing_files[0]["id"]
        f = drive.CreateFile(
            {"parents": {"id": folder_id}, "id": id, "title": f["title"]}
        )
        f.SetContentFile(path)
        f.Upload()
    else:
        # File does not exist, upload it
        f.Upload()


if __name__ == "__main__":
    drive = g_auth()
    g_uploder(drive, r"C:\Users\Inori\a_pycrypt\DB_maker\IRBANK_CODE_0.csv", "test")
