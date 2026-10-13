set -e
git config --system --add safe.directory "*"
export GIT_AUTHOR_NAME=pat GIT_AUTHOR_EMAIL=pat@home GIT_COMMITTER_NAME=pat GIT_COMMITTER_EMAIL=pat@home
export GIT_AUTHOR_DATE="2026-08-01T10:00:00" GIT_COMMITTER_DATE="2026-08-01T10:00:00"
mkdir -p /home/pat/budget && cd /home/pat/budget
git init -q -b main
echo "month,item,amount" > budget.csv
git add . && git commit -qm "Start budget"
echo "2026-06,rent,1200" >> budget.csv; git commit -qam "Add June rent"
echo "2026-07,rent,1200" >> budget.csv; git commit -qam "Add July rent"
git checkout -q --detach
echo "2026-08,rent,1250" >> budget.csv; git commit -qam "Add August rent (increase)"
echo "vacation fund: 300 per month" > notes.md; git add notes.md; git commit -qm "Add vacation fund note"
chown -R pat:pat /home/pat
