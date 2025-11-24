# sepa_python

This folder contains various python functions / tools in development.

To use them in Spyder environment:
*   Open Tools -> PYTHONPATH manager
*   click + icon at right side of window
*	add folder where this file is located
*	restart Spyder

You should now be able to access them all. There are better ways of managing this (e.g. pip). Will sort one day!

The contents of this folder can be inspected using

`import sepa_tools`

`sepa_tools.string_tools.utils.show_package_tree('C:\Python\sepa_python')`


**scripts** - templates for doing various processing tasks

**mike_tools** - python package containing general tools for processing Mike files (currently only mesh_index function!)

**part_processing** - python package for processing particle tracking output, including

	* converting xml to mat (raw particle tracks)
	* generating surface concentrations
	* calculating exposure along fish tracks
	
**sepa_tools** - python package with various tools 

	* dict_tools - filter, compare, display sizes, add items
	* string_tools - string_finder, file_finder, closest_string_match
	* load_mat_file - create python dict from .mat file
