cd /home/pat/contacts
iconv -f latin1 -t utf-8 customers_merged.csv | tr -d '\r' | sed '1s/^\xc3\xaf\xc2\xbb\xc2\xbf//' > customers_clean.csv
