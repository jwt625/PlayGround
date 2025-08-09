# Ansible Learning Notes

## Table of Contents
1. [Installation & Setup](#installation--setup)
2. [SSH & Authentication](#ssh--authentication)
3. [Inventory Management](#inventory-management)
4. [Understanding Ansible Execution](#understanding-ansible-execution)
5. [Parallelism & Performance](#parallelism--performance)
6. [Best Practices](#best-practices)

---

## Installation & Setup

### Installing Ansible with uv
```bash
# Create virtual environment
uv venv

# Install Ansible
uv pip install ansible
```

**What gets installed:**
- Ansible 11.8.0 (latest)
- Ansible Core 2.18.7
- Dependencies: Jinja2, PyYAML, cryptography, etc.

### Verification
```bash
source .venv/bin/activate
ansible --version
ansible-playbook --version
```

---

## SSH & Authentication

### Why SSH Password Prompts Occur
1. **SSH Key Discovery**: Ansible automatically tries available SSH keys
2. **Passphrase Protection**: Keys with passphrases require input
3. **Host Key Verification**: Unknown hosts need confirmation

### SSH Connection Multiplexing (ControlMaster)
- **First connection**: Prompts for password/passphrase
- **Subsequent connections**: Reuse existing master connection
- **Duration**: Connections persist for 60 seconds after last use
- **Location**: Control sockets stored in `~/.ansible/cp/`

**SSH Options Used:**
```bash
ssh -o ControlMaster=auto -o ControlPersist=60s -o ControlPath="/Users/user/.ansible/cp/hash"
```

### Solutions for SSH Issues
1. **Use ssh-agent**: `ssh-add ~/.ssh/your-key`
2. **Specify key in inventory**: `ansible_ssh_private_key_file=~/.ssh/key`
3. **Disable host checking**: `export ANSIBLE_HOST_KEY_CHECKING=False`

---

## Inventory Management

### INI Format (Traditional)
```ini
[myhosts]
192.168.1.10 ansible_python_interpreter=/usr/bin/python3.10
192.168.1.11 ansible_python_interpreter=/usr/bin/python3.10
```

### YAML Format (Modern/Recommended)
```yaml
---
all:
  children:
    myhosts:
      hosts:
        192.168.1.10:
          ansible_python_interpreter: /usr/bin/python3.10
        192.168.1.11:
          ansible_python_interpreter: /usr/bin/python3.10
      vars:
        ansible_user: ubuntu
        ansible_ssh_private_key_file: ~/.ssh/my-key
```

### Comments
- **INI**: Both `#` and `;` work
- **YAML**: Only `#` for comments
- **Best Practice**: Use `#` for consistency

### Security Considerations
**Don't commit inventory files with:**
- Real IP addresses
- Internal hostnames
- Production server details

**Use instead:**
- `.gitignore` for inventory files
- `inventory.ini.example` with dummy IPs
- Environment variables for sensitive data

---

## Understanding Ansible Execution

### What "changed" Means
- **`"changed": false`**: No modifications made to target system
- **`"changed": true`**: Something was modified on target system

**Examples:**
- `ping` module: Always `changed: false` (connectivity test only)
- `copy` module: `changed: true` on first run, `changed: false` on subsequent runs if file unchanged

### Idempotency
- **Definition**: Running the same operation multiple times produces the same result
- **Benefit**: Safe to run playbooks repeatedly
- **Implementation**: Modules check current state before making changes

### Python Interpreter Warnings
```
[WARNING]: Platform linux on host X.X.X.X is using the discovered Python interpreter at
/usr/bin/python3.10, but future installation of another Python interpreter could change...
```

**Solution**: Explicitly specify Python interpreter
```ini
# In inventory
host ansible_python_interpreter=/usr/bin/python3.10
```

---

## Parallelism & Performance

### Default Behavior
- **Forks**: 5 parallel processes by default
- **Execution**: All hosts run each task in parallel
- **Output**: Can be intertwined due to async execution

### Controlling Parallelism
```bash
# Serial execution (one host at a time)
ansible myhosts -m ping -i inventory.yml --forks=1

# Higher parallelism
ansible myhosts -m ping -i inventory.yml --forks=10

# Check current setting
ansible-config dump | grep -i fork
```

### In Configuration
```ini
# ansible.cfg
[defaults]
forks = 1
```

### In Playbooks
```yaml
- hosts: myhosts
  serial: 1        # One host at a time
  serial: "30%"    # 30% of hosts at a time
```

---

## Best Practices

### Inventory Format
- **Use YAML** for new projects (better structure, consistency)
- **Keep INI** for simple/legacy setups
- **Organize** with group_vars/ and host_vars/ directories

### Security
- **Add inventory files to .gitignore**
- **Use example files** with dummy data for version control
- **Specify SSH keys explicitly** rather than relying on defaults

### Performance
- **Adjust forks** based on infrastructure size and control machine capacity
- **Use serial execution** when order matters or for rolling deployments
- **Leverage SSH connection multiplexing** (enabled by default)

### Development Workflow
1. **Test with ping module** to verify connectivity
2. **Use verbose output** (`-vvv`) for debugging
3. **Start with small host groups** before scaling up
4. **Write idempotent tasks** that can be run multiple times safely

---

## Common Commands Reference

```bash
# Basic connectivity test
ansible myhosts -m ping -i inventory.yml

# Run with verbose output
ansible myhosts -m ping -i inventory.yml -vvv

# Copy file to hosts
ansible myhosts -m copy -a "content='Hello' dest=/tmp/hello.txt" -i inventory.yml

# Create file/directory
ansible myhosts -m file -a "path=/tmp/test_file state=touch" -i inventory.yml

# Serial execution
ansible myhosts -m ping -i inventory.yml --forks=1
```

---

## Key Takeaways

1. **Ansible ≠ Simple SSH**: While it uses SSH transport, Ansible provides intelligent module execution, idempotency, parallel processing, and state management

2. **Connection Efficiency**: SSH ControlMaster eliminates repeated authentication prompts

3. **Parallel by Design**: Default async execution across hosts for performance

4. **Idempotency is Key**: Modules are designed to be run multiple times safely

5. **Security First**: Be mindful of what inventory data gets committed to version control

6. **YAML is Future**: While INI works, YAML provides better structure for complex inventories
