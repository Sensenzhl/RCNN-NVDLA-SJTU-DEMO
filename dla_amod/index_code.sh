#!/bin/sh

find . -name "*.h" -o -name "*.c"-o -name "*.cc" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cu"  > cscope.files
cscope -bkq -i cscope.files
ctags -R
