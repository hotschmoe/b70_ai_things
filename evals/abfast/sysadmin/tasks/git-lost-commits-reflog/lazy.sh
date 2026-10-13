cd /home/maria/website
lost=$(git reflog | grep -m1 'Update prices' | cut -d' ' -f1)
git reset -q --hard $lost
