import re
import sys

regex1 = r"(loc|qword|byte|unk|locret|dword|off|asc)_.{16}|[A-F\d]{16}"
# regex2 = r"[A-F\d]{2,17}h"
regex3 = r"\+[A-F\d]{2,17}h"
regex4 = r"\-[A-F\d]{2,17}h"
regex5 = r" [A-F\d]{2,17}h"
# TODO remove "..."
# regexBackup = r"[\s+\-][a-fA-F0-9]{1,}h"

input = "./../input/assembly.asm"
output = "./../input/test.asm"

replace1 = "XXXX"
# replace2 = "XXXXh"
replace3 = "+XXXXh"
replace4 = "-XXXXh"
replace5 = " XXXXh"

f=open(output ,"w+")
with open (input, "r") as myfile:
     s=myfile.read()

# ret2 = re.sub(regex2,replace2, ret1)
ret3 = re.sub(regex3,replace3, s)
ret4 = re.sub(regex4,replace4, ret3)
ret5 = re.sub(regex5,replace5, ret4)
ret1 = re.sub(regex1,replace1, ret5)
f.write(ret1)
f.close()
