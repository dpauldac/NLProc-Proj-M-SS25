# README: Installing Visual Studio Code

This guide provides instructions on how to install Visual Studio Code (VS Code), a lightweight but powerful source code editor which runs on your desktop and is available for Windows, macOS and Linux.

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Installation Steps](#installation-steps)
    * [Windows](#windows)
    * [macOS](#macos)
    * [Linux](#linux)
        * [Debian/Ubuntu based distributions](#debianubuntu-based-distributions)
        * [RHEL/Fedora/SUSE based distributions](#rhelfedoraSUSE-based-distributions)
        * [Snap](#snap)
        * [Arch Linux](#arch-linux)
3.  [Verifying the Installation](#verifying-the-installation)
4.  [Launching VS Code](#launching-vs-code)
5.  [Troubleshooting](#troubleshooting)
6.  [Further Information](#further-information)

## Prerequisites

* **Operating System:**
    * Windows 10 or 11 (64-bit recommended)
    * macOS 10.15 (Catalina) or newer (Universal binary for Intel and Apple Silicon)
    * Linux:
        * Debian/Ubuntu based: Ubuntu 20.04, Debian 10 or newer
        * RHEL/Fedora/SUSE based: RHEL 7, Fedora 34, openSUSE 15.2 or newer
        * GLIBCXX version 3.4.25 or later
        * GLIBC version 2.28 or later
* **Hardware (Minimum):**
    * 1.6 GHz or faster processor
    * 1 GB of RAM
    * ~250 MB of disk space (plus space for user data and extensions)
* **Internet Connection:** Required for downloading the installer and for some features/extensions.

## Installation Steps

### Windows

1.  **Download the Installer:**
    * Go to the official VS Code download page: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
    * Download the **Windows User Installer** (recommended for most users) or the System Installer (requires administrator privileges). Typically, the `64-bit` version is appropriate for modern systems.

2.  **Run the Installer:**
    * Once the download is complete, locate the `.exe` file (e.g., `VSCodeUserSetup-x64-<version>.exe`) in your Downloads folder and double-click it to run.
    * Accept the license agreement and click **Next**.

3.  **Select Destination Location:**
    * Choose the installation directory. The default location is usually fine. Click **Next**.

4.  **Select Start Menu Folder:**
    * Choose the Start Menu folder name. The default is usually fine. Click **Next**.

5.  **Select Additional Tasks:**
    * You'll be presented with options such as:
        * **Create a desktop icon:** Check this if you want a shortcut on your desktop.
        * **Add "Open with Code" action to Windows Explorer file context menu:** Allows you to right-click a file and open it in VS Code.
        * **Add "Open with Code" action to Windows Explorer directory context menu:** Allows you to right-click a folder and open it as a project in VS Code.
        * **Register Code as an editor for supported file types:** Makes VS Code the default editor for certain file types.
        * **Add to PATH (requires shell restart):** This is highly recommended as it allows you to launch `code` from the command line.
    * Select the options you prefer and click **Next**.

6.  **Install:**
    * Click **Install** to begin the installation process.

7.  **Finish:**
    * Once the installation is complete, you can choose to launch Visual Studio Code immediately. Click **Finish**.

### macOS

1.  **Download the Application:**
    * Go to the official VS Code download page: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
    * Download the **Mac Universal** build (supports both Intel and Apple Silicon Macs). This will download a `.zip` file.

2.  **Extract the Application:**
    * Once the download is complete, locate the `Visual Studio Code.app.zip` file (or similar) in your Downloads folder.
    * Double-click the `.zip` file to extract the application. This will create a `Visual Studio Code.app` file.

3.  **Move to Applications Folder:**
    * Drag the `Visual Studio Code.app` file to your `/Applications` folder. This makes it available in Launchpad and Finder.

4.  **Add to Dock (Optional):**
    * Open the Applications folder, find Visual Studio Code, and drag its icon to the Dock for quick access.

5.  **Add 'code' command to PATH (Optional but Recommended):**
    * Open VS Code.
    * Open the Command Palette using `Cmd+Shift+P` (or `View > Command Palette...`).
    * Type `Shell Command` and select **Shell Command: Install 'code' command in PATH**.
    * You might be prompted for your administrator password.
    * After this, you can open files and folders in VS Code from your terminal using the `code` command (e.g., `code .` to open the current directory).

### Linux

VS Code can be installed in several ways on Linux distributions.

#### Debian/Ubuntu based distributions

1.  **Download the .deb package:**
    * Go to the official VS Code download page: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
    * Download the `.deb` package (`64-bit`).

2.  **Install via the command line (recommended):**
    * Open your terminal.
    * Navigate to the directory where you downloaded the file (e.g., `cd ~/Downloads`).
    * Install the package using `apt` (this will also install dependencies):
        ```bash
        sudo apt update
        sudo apt install ./<filename>.deb
        ```
        Replace `<filename>.deb` with the actual name of the downloaded file (e.g., `code_1.89.0-1680091916_amd64.deb`).
    * If you encounter any dependency issues, run:
        ```bash
        sudo apt -f install
        ```

3.  **Install via GUI (Software Center):**
    * Alternatively, you can often double-click the downloaded `.deb` file to open it with your distribution's Software Center and install it from there.

#### RHEL/Fedora/SUSE based distributions

1.  **Download the .rpm package:**
    * Go to the official VS Code download page: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
    * Download the `.rpm` package (`64-bit`).

2.  **Install via the command line (recommended):**
    * Open your terminal.
    * Navigate to the directory where you downloaded the file (e.g., `cd ~/Downloads`).
    * Install the package:
        * For RHEL/CentOS/Rocky Linux:
            ```bash
            sudo yum install ./<filename>.rpm
            ```
            or if using `dnf`:
            ```bash
            sudo dnf install ./<filename>.rpm
            ```
        * For Fedora:
            ```bash
            sudo dnf install ./<filename>.rpm
            ```
        * For openSUSE:
            ```bash
            sudo zypper install ./<filename>.rpm
            ```
        Replace `<filename>.rpm` with the actual name of the downloaded file.

3.  **Alternatively, use the Microsoft repository (recommended for automatic updates):**
    * **For RHEL, Fedora, CentOS, Rocky Linux:**
        ```bash
        sudo rpm --import [https://packages.microsoft.com/keys/microsoft.asc](https://packages.microsoft.com/keys/microsoft.asc)
        sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=[https://packages.microsoft.com/yumrepos/vscode](https://packages.microsoft.com/yumrepos/vscode)\nenabled=1\ngpgcheck=1\ngpgkey=[https://packages.microsoft.com/keys/microsoft.asc](https://packages.microsoft.com/keys/microsoft.asc)" > /etc/yum.repos.d/vscode.repo'
        sudo dnf check-update # or sudo yum check-update
        sudo dnf install code # or sudo yum install code
        ```
    * **For openSUSE and SLE:**
        ```bash
        sudo rpm --import [https://packages.microsoft.com/keys/microsoft.asc](https://packages.microsoft.com/keys/microsoft.asc)
        sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=[https://packages.microsoft.com/yumrepos/vscode](https://packages.microsoft.com/yumrepos/vscode)\nenabled=1\ntype=rpm-md\ngpgcheck=1\ngpgkey=[https://packages.microsoft.com/keys/microsoft.asc](https://packages.microsoft.com/keys/microsoft.asc)" > /etc/zypp/repos.d/vscode.repo'
        sudo zypper refresh
        sudo zypper install code
        ```

#### Snap

VS Code is available as a Snap package. If you have Snap installed and configured on your system:

1.  Open your terminal.
2.  Install VS Code using Snap:
    ```bash
    sudo snap install --classic code
    ```
    Snaps update automatically.

#### Arch Linux

VS Code is available in the Arch User Repository (AUR) or the official community repository.

1.  **From the community repository (recommended):**
    * Open your terminal.
    * Update your package database and install `code`:
        ```bash
        sudo pacman -Syu code
        ```

2.  **From the AUR (using an AUR helper like `yay` or `paru`):**
    * If you want the `visual-studio-code-bin` (pre-compiled binary) package:
        ```bash
        yay -S visual-studio-code-bin
        ```
        or
        ```bash
        paru -S visual-studio-code-bin
        ```
    * Follow the prompts from your AUR helper.

## Verifying the Installation

You can verify the installation by:

1.  **Launching VS Code** (see next section).
2.  **Checking the version from the command line** (if added to PATH or installed via package manager):
    * Open a new terminal or command prompt.
    * Type the following command and press Enter:
        ```bash
        code --version
        ```
    * This should output the installed version of VS Code.

## Launching VS Code

* **Windows:**
    * Click the Start button, search for "Visual Studio Code", and click on it.
    * If you created a desktop icon, double-click it.
    * From the command prompt or PowerShell (if added to PATH): `code`

* **macOS:**
    * Open Launchpad and click on the Visual Studio Code icon.
    * Open Finder, go to the Applications folder, and double-click Visual Studio Code.
    * From the terminal (if 'code' command installed in PATH): `code`

* **Linux:**
    * Open your applications menu and search for "Visual Studio Code" or "Code" and click on it.
    * From the terminal: `code`

## Troubleshooting

* **"Command 'code' not found" (after installing and adding to PATH):**
    * **Windows:** You may need to restart your terminal session, or even log out and log back in, or restart your computer for the PATH changes to take effect.
    * **macOS/Linux:** Ensure you ran the "Install 'code' command in PATH" from the Command Palette (macOS) or that your shell's configuration file (`.bashrc`, `.zshrc`, etc.) correctly sources the path if installed manually. A new terminal session is usually required.

* **Installation fails due to missing dependencies (Linux):**
    * If using `.deb` or `.rpm` packages directly, ensure your system is up to date (`sudo apt update` or `sudo dnf check-update`).
    * For `.deb` packages, `sudo apt -f install` can often resolve dependency issues.
    * Refer to the official VS Code documentation for specific dependency requirements for your distribution if problems persist.

* **Permissions issues (Linux):**
    * Ensure you are using `sudo` when installing via package managers or when modifying system directories.

* **"App is damaged and can't be opened" (macOS):**
    * This can sometimes happen due to Gatekeeper settings or a corrupted download.
    * Try re-downloading the `.zip` file.
    * Ensure you're moving the `.app` file to the `/Applications` folder.
    * If the issue persists, you might need to temporarily adjust your security settings in "System Settings" > "Privacy & Security" (be cautious and revert after installation).

## Further Information

* **Official VS Code Documentation:** [https://code.visualstudio.com/docs](https://code.visualstudio.com/docs)
* **VS Code Download Page:** [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
* **VS Code Setup Overview:** [https://code.visualstudio.com/docs/setup/setup-overview](https://code.visualstudio.com/docs/setup/setup-overview)

---

Happy Coding!