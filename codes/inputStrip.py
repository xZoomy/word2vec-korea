import re
import sys

regex1 = r"\+[A-F\d]{2,17}h"
regex2 = r"\-[A-F\d]{2,17}h"
regex3 = r" [A-F\d]{2,17}h"
regex4 = r"(loc|qword|byte|unk|locret|dword|off|asc)_[^\s]{16}\b|[A-F\d]{16}"
regex5 = r";[^\n]{0,}"
# regexBackup = r"[\s+\-][a-fA-F0-9]{1,}h"

input = "./../input/assembly.asm"
output = "./../input/test.asm"

replace1 = "+XXXXh"
replace2 = "-XXXXh"
replace3 = " XXXXh"
replace4 = "XXXX"
replace5 = " "

f=open(output ,"w+")
with open (input, "r") as myfile:
     s=myfile.read()

ret1 = re.sub(regex1,replace1, s)
ret2 = re.sub(regex2,replace2, ret1)
ret3 = re.sub(regex3,replace3, ret2)
ret4 = re.sub(regex4,replace4, ret3)
ret5 = re.sub(regex5, replace5, ret4)
f.write(ret5)
f.close()
