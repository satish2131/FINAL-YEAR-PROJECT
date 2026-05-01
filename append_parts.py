import re
import sys

path = r"c:\Users\dsaiv\OneDrive\Desktop\final year project\Latent_Facial_Identity_Mapping\facial_identity_interactive_parts.html"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# EYES
text = re.sub(
    r"(name: 'Wide-set', draw: s => \{\s*s\.querySelector\('#eye-group-l'\)\.setAttribute\('transform', 'translate\(-14, 0\)'\);\s*s\.querySelector\('#eye-group-r'\)\.setAttribute\('transform', 'translate\(14, 0\)'\);\s*s\.querySelectorAll\('#eyes-group ellipse'\)\.forEach\(e => \{ e\.setAttribute\('ry', '13'\); e\.setAttribute\('rx', '24'\) \}\);\s*\}\s*\},)(\s*\])",
    r"\1\n        {\n          name: 'Actor Gaze', draw: s => {\n            s.querySelector('#eye-group-l').setAttribute('transform', 'none');\n            s.querySelector('#eye-group-r').setAttribute('transform', 'none');\n            s.querySelectorAll('#eyes-group ellipse').forEach(e => { e.setAttribute('ry', '11'); e.setAttribute('rx', '22') });\n          }\n        },\2",
    text
)

# MOUTH
text = re.sub(
    r"(name: 'Full lips', draw: s => \{\s*s\.querySelector\('#mouth-top'\)\.setAttribute\('d', 'M125 258 Q160 244 195 258'\);\s*s\.querySelector\('#mouth-bot'\)\.setAttribute\('d', 'M125 258 Q160 288 195 258'\);\s*\}\s*\},)(\s*\])",
    r"\1\n        {\n          name: 'Actor Smile', draw: s => {\n            s.querySelector('#mouth-top').setAttribute('d', 'M120 255 Q160 240 200 255');\n            s.querySelector('#mouth-bot').setAttribute('d', 'M120 255 Q160 300 200 255');\n          }\n        },\2",
    text
)

# HAIR
text = re.sub(
    r"(name: 'Bald', draw: s => \{\s*s\.querySelector\('#hair'\)\.setAttribute\('d', 'M0 0'\);\s*\}\s*\},)(\s*\])",
    r"\1\n        { name: 'Actor Swept', draw: s => { s.querySelector('#hair').setAttribute('d', 'M50 190 Q40 110 80 60 Q120 20 180 40 Q240 50 270 100 Q280 150 265 190 Q240 150 220 140 Q190 120 160 130 Q130 110 100 140 Q80 160 50 190 Z'); s.querySelector('#hair').setAttribute('fill', '#1a1717'); } },\2",
    text
)

# BEARD
text = re.sub(
    r"(name: 'Goatee', draw: s => \{\s*s\.querySelector\('#beard-standalone-group'\)\.innerHTML =\s*'<path d=\"M140 265 Q160 340 180 265 Q160 355 140 265Z\" fill=\"#1e1616\" opacity=\"0\.88\" filter=\"url\(#shadow\)\"/><path d=\"M142 263 Q160 255 178 263 Q160 272 142 263Z\" fill=\"#1e1616\" opacity=\"0\.9\"/>';\s*\}\s*\},)(\s*\])",
    r"\1\n        {\n          name: 'Actor Thick', draw: s => {\n            s.querySelector('#beard-standalone-group').innerHTML =\n              '<path d=\"M78 245 Q160 380 242 245 Q200 320 160 330 Q120 320 78 245Z\" fill=\"#151010\" opacity=\"0.95\" filter=\"url(#shadow)\"/><path d=\"M115 240 Q160 220 205 240 Q160 275 115 240Z\" fill=\"#151010\" opacity=\"0.95\"/><path d=\"M70 175 Q80 220 85 255 L65 240Z\" fill=\"#151010\" opacity=\"0.95\"/><path d=\"M250 175 Q240 220 235 255 L255 240Z\" fill=\"#151010\" opacity=\"0.95\"/>';\n          }\n        },\2",
    text
)

# MUSTACHE
text = re.sub(
    r"(name: 'Handlebar', draw: s => \{\s*s\.querySelector\('#mustache-group'\)\.innerHTML =\s*'<path[^\>]*>.*?';\s*\}\s*\},)(\s*\])",
    r"\1\n        {\n          name: 'Actor Stache', draw: s => {\n            s.querySelector('#mustache-group').innerHTML =\n              '<path d=\"M115 255 Q160 240 205 255 Q180 265 160 270 Q140 265 115 255Z\" fill=\"#151010\" opacity=\"0.95\"/>';\n          }\n        },\2",
    text,
    flags=re.DOTALL
)

# Replace DEFS
# Eyes def
text = text.replace(
    "'<ellipse cx=\"15\" cy=\"22\" rx=\"12\" ry=\"9\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.1)\"/><circle cx=\"15\" cy=\"22\" r=\"5\" fill=\"#00f0ff\" opacity=\".6\"/><ellipse cx=\"61\" cy=\"22\" rx=\"12\" ry=\"9\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.1)\"/><circle cx=\"61\" cy=\"22\" r=\"5\" fill=\"#00f0ff\" opacity=\".6\"/>'],",
    "'<ellipse cx=\"15\" cy=\"22\" rx=\"12\" ry=\"9\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.1)\"/><circle cx=\"15\" cy=\"22\" r=\"5\" fill=\"#00f0ff\" opacity=\".6\"/><ellipse cx=\"61\" cy=\"22\" rx=\"12\" ry=\"9\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.1)\"/><circle cx=\"61\" cy=\"22\" r=\"5\" fill=\"#00f0ff\" opacity=\".6\"/>',\n        '<ellipse cx=\"22\" cy=\"22\" rx=\"14\" ry=\"7\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.1)\"/><circle cx=\"22\" cy=\"22\" r=\"4.5\" fill=\"#00f0ff\" opacity=\".6\"/><ellipse cx=\"54\" cy=\"22\" rx=\"14\" ry=\"7\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.1)\"/><circle cx=\"54\" cy=\"22\" r=\"4.5\" fill=\"#00f0ff\" opacity=\".6\"/>'],"
)

# Mouth def
text = text.replace(
    "'<path d=\"M10 16 Q38 4 66 16\" stroke=\"#00f0ff\" stroke-width=\"1\" fill=\"none\" stroke-linecap=\"round\"/><path d=\"M10 16 Q38 38 66 16\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.07)\" stroke-linecap=\"round\"/>'],",
    "'<path d=\"M10 16 Q38 4 66 16\" stroke=\"#00f0ff\" stroke-width=\"1\" fill=\"none\" stroke-linecap=\"round\"/><path d=\"M10 16 Q38 38 66 16\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.07)\" stroke-linecap=\"round\"/>',\n        '<path d=\"M8 12 Q38 0 68 12\" stroke=\"#00f0ff\" stroke-width=\"1\" fill=\"none\" stroke-linecap=\"round\"/><path d=\"M8 12 Q38 42 68 12\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"rgba(0,240,255,0.07)\" stroke-linecap=\"round\"/>'],"
)

# Hair def
text = text.replace(
    "'<path d=\"M4 28 Q38 22 72 28\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"none\" stroke-linecap=\"round\"/>'],",
    "'<path d=\"M4 28 Q38 22 72 28\" stroke=\"#00f0ff\" stroke-width=\"1.5\" fill=\"none\" stroke-linecap=\"round\"/>',\n        '<path d=\"M4 28 Q0 5 38 4 Q76 8 72 28 C64 20 60 18 52 22 C44 14 38 20 32 14 C24 22 20 18 4 28Z\" stroke=\"#111\" stroke-width=\"1.5\" fill=\"#222\" stroke-linecap=\"round\"/>'],"
)

# Beard def
text = text.replace(
    "'<path d=\"M30 20 Q38 42 46 20 Q38 46 30 20Z\" fill=\"#1e1616\" opacity=\"0.9\"/><path d=\"M28 18 Q38 12 48 18 Q38 24 28 18Z\" fill=\"#1e1616\" opacity=\"0.9\"/>'",
    "'<path d=\"M30 20 Q38 42 46 20 Q38 46 30 20Z\" fill=\"#1e1616\" opacity=\"0.9\"/><path d=\"M28 18 Q38 12 48 18 Q38 24 28 18Z\" fill=\"#1e1616\" opacity=\"0.9\"/>',\n        '<path d=\"M2 14 Q38 46 74 14 Q50 36 38 40 Q26 36 2 14Z\" fill=\"#1a1212\" opacity=\"0.95\"/><path d=\"M16 12 Q38 4 60 12 Q38 18 16 12Z\" fill=\"#1a1212\" opacity=\"0.95\"/>'"
)

# Mustache def
text = text.replace(
    "'<path d=\"M16 18 Q28 10 38 16 Q48 10 60 18 Q50 22 38 18 Q26 22 16 18Z\" fill=\"#1a1212\" opacity=\"0.9\"/><path d=\"M16 18 Q6 14 4 6 Q10 14 16 18Z\" fill=\"#1a1212\" opacity=\"0.9\"/><path d=\"M60 18 Q70 14 72 6 Q66 14 60 18Z\" fill=\"#1a1212\" opacity=\"0.9\"/>'",
    "'<path d=\"M16 18 Q28 10 38 16 Q48 10 60 18 Q50 22 38 18 Q26 22 16 18Z\" fill=\"#1a1212\" opacity=\"0.9\"/><path d=\"M16 18 Q6 14 4 6 Q10 14 16 18Z\" fill=\"#1a1212\" opacity=\"0.9\"/><path d=\"M60 18 Q70 14 72 6 Q66 14 60 18Z\" fill=\"#1a1212\" opacity=\"0.9\"/>',\n        '<path d=\"M10 18 Q28 6 38 12 Q48 6 66 18 Q52 24 38 28 Q24 24 10 18Z\" fill=\"#1a1212\" opacity=\"0.95\"/>'"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Updates applied successfully.")
