for %%f in (output/*.csv) do (
	echo Generate graph for %%f
	python draw-grapth2.py ./output/%%f 
)
