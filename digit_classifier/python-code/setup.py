# setup.py
#hide
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()