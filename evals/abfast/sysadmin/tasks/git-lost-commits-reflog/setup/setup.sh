set -e
git config --system --add safe.directory "*"
export GIT_AUTHOR_NAME=maria GIT_AUTHOR_EMAIL=maria@shop GIT_COMMITTER_NAME=maria GIT_COMMITTER_EMAIL=maria@shop
export GIT_AUTHOR_DATE="2026-10-01T10:00:00" GIT_COMMITTER_DATE="2026-10-01T10:00:00"
mkdir -p /home/maria/website && cd /home/maria/website
git init -q -b main
echo "<h1>Maria's shop</h1>" > index.html; echo "Home" > about.txt; git add .; git commit -qm "Initial site"
echo "<p>Opening hours</p>" >> index.html; git commit -qam "Add opening hours"
git branch staging
git checkout -q staging; echo "beta banner" > banner.txt; git add .; git commit -qm "Staging banner"; git checkout -q main
echo "<a href=contact.html>Contact</a>" >> index.html; echo "<h1>Contact</h1>" > contact.html; git add .; git commit -qm "Add contact page"
echo "<footer>(c) 2026</footer>" > footer.html; git add .; git commit -qm "Fix footer"
echo "cake 20" > prices.txt; git add .; git commit -qm "Update prices"
git reset -q --hard HEAD~3
echo "Hom" > about.txt; echo "Home page" > about.txt; git commit -qam "Fix typo in about"
git reflog expire --expire=never --all
chown -R maria:maria /home/maria
