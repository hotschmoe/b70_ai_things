import random
FIRST = ["Jos\u00e9", "Fran\u00e7ois", "Zo\u00eb", "S\u00f8ren", "Ana\u00efs", "Bj\u00f6rn", "Chlo\u00e9", "Andr\u00e9", "Ren\u00e9e", "Ma\u00ebl", "Lu\u00eds", "\u00c9mile"]
LAST = ["M\u00fcller", "Garc\u00eda", "Dubois", "N\u00fa\u00f1ez", "\u00d8stergaard", "Sch\u00e4fer", "Lef\u00e8vre", "Mu\u00f1oz", "Kowalski", "O'Brien", "Caf\u00e9r", "\u00c5berg"]
COMP = ["Caf\u00e9 \u20acuro", "Bistro \u2019Ouest\u2019", "Boulangerie \u201cLa Mie\u201d", "Pastelaria Lu\u00eds & Filhos", "Tech \u20ac GmbH", "Sch\u00f6n Bau AG"]
def rows():
    r = random.Random(7); out = []
    for i in range(1, 101):
        out.append((i, r.choice(FIRST), r.choice(LAST), r.choice(COMP)))
    return out
