set -e
python3 -m venv --clear /opt/reports/venv
/opt/reports/venv/bin/pip install --no-index --find-links /opt/wheelhouse -r /opt/reports/requirements.txt
