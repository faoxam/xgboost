@echo off
call :treeProcess
goto :eof

:treeProcess
rem Do whatever you want here over the files of this subdir, for example:
for %%f in (input/*.csv) do (
	echo execute calculation with searies %%f 
	python predict.py %%f DEPTH GR
	python predict.py %%f DEPTH CALI
	python predict.py %%f DEPTH NPHI
	python predict.py %%f DEPTH RHOB
	python predict.py %%f DEPTH RSHAL
	python predict.py %%f DEPTH RDEEP
	python predict.py %%f DEPTH RXO
	python predict.py %%f DEPTH SP
	python predict.py %%f DEPTH SW
	
	echo Archiving %%f 
	copy input\%%f archive\%%f
	del input\%%f
)
