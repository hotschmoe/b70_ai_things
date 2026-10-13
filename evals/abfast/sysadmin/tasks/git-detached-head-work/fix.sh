cd /home/pat/budget
export GIT_COMMITTER_NAME=pat GIT_COMMITTER_EMAIL=pat@home GIT_AUTHOR_NAME=pat GIT_AUTHOR_EMAIL=pat@home
git branch tmp-rescue
git checkout -q main
git merge -q --ff-only tmp-rescue
git branch -D tmp-rescue
