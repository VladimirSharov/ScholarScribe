import subprocess
import datetime

def run_command(command):
    """Run a command using subprocess and return the output, capturing stderr."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stderr:
            print("Command executed but returned an error:", result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running command:", str(e))
        print("Error output:", e.stderr)
        return None

def update_development_log():
    """Append a new entry to the project_journal.txt."""
    print("Updating project_journal.txt...")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"\n## Update on {timestamp}\n- Finalized session updates.\n"
    with open('project_journal.txt', 'a') as file:
        file.write(log_message)

def update_changelog():
    """Automatically generate a changelog entry."""
    print("Updating changelog.txt...")
    changelog_message = f"Changes on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n"
    git_status = run_command("git status --porcelain")
    if git_status:
        changes = git_status.strip().split('\n')
        for change in changes:
            change_type, file_path = change[0:2], change[3:]
            if change_type == '??':
                changelog_message += f"- Added {file_path}\n"
            else:
                changelog_message += f"- Modified {file_path}\n"
    with open('changelog.txt', 'a') as file:
        file.write(changelog_message)

def git_operations():
    """Add changes, commit, and push to the remote git repository."""
    print("Adding changes to git...")
    run_command("git add .")
    
    print("Committing changes...")
    commit_message = "Update session: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    run_command(f"git commit -m \"{commit_message}\"")
    
    print("Pushing changes to GitHub...")
    return run_command("git push origin master")

if __name__ == "__main__":
    print("Please ensure to manually update 'requirements-frozen.txt' if any dependencies have changed.")
    update_development_log()
    update_changelog()
    if git_operations():
        print("Successfully updated and pushed to GitHub.")
    else:
        print("Failed to push changes to GitHub.")
