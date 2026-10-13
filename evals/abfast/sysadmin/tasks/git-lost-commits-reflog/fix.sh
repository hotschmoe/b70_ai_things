cd /home/maria/website
export GIT_COMMITTER_NAME=maria GIT_COMMITTER_EMAIL=maria@shop GIT_AUTHOR_NAME=maria GIT_AUTHOR_EMAIL=maria@shop
lost=$(git reflog | grep -m1 'Update prices' | cut -d' ' -f1)
git merge -q --no-edit $lost
