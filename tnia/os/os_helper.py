import platform

def is_linux():
    """
    Detects if the current system is Linux.

    Returns:
        bool: True if the system is Linux, False otherwise.
    """
    return platform.system().lower() == "linux"

def convert_path_for_wsl(path):
    """
    Converts a Windows path to a WSL path if running inside WSL2.
    If not in WSL2 or if not Linux, returns the original path.

    Args:
        path (str): The original Windows path.

    Returns:
        str: The converted path for WSL2 or the original path for Windows.
    """
    if is_linux():
        # Detect if running inside WSL2
        if "microsoft" in platform.uname().release.lower():
            # Convert Windows path to WSL2 format
            #drive, rest = os.path.splitdrive(path)
            #drive_letter = drive.rstrip(":").lower()
            
            #drive_letter = path.split(':')[0].lower()
            #rest = path.split(':')[1]
            #converted_path = f"/mnt/{drive_letter}{rest.replace("\\", /')}"
            #return converted_path

            drive_letter, rest = path.split(":", 1)
            drive_letter = drive_letter.lower()
            rest = rest.replace("\\", "/")  # Replace backslashes with forward slashes
            converted_path = f"/mnt/{drive_letter}{rest}"
            return converted_path
    return path  # Return the original path if not Linux or not WSL2