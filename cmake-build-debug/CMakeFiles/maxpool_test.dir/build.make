# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\Program Files\JetBrains\CLion 2021.3.3\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\Program Files\JetBrains\CLion 2021.3.3\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\malav\CLionProjects\ANN-master\ANN-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\malav\CLionProjects\ANN-master\ANN-master\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/maxpool_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/maxpool_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/maxpool_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/maxpool_test.dir/flags.make

CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.obj: CMakeFiles/maxpool_test.dir/flags.make
CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.obj: CMakeFiles/maxpool_test.dir/includes_CXX.rsp
CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.obj: ../testing/maxpool_test.cpp
CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.obj: CMakeFiles/maxpool_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\malav\CLionProjects\ANN-master\ANN-master\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.obj"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.obj -MF CMakeFiles\maxpool_test.dir\testing\maxpool_test.cpp.obj.d -o CMakeFiles\maxpool_test.dir\testing\maxpool_test.cpp.obj -c C:\Users\malav\CLionProjects\ANN-master\ANN-master\testing\maxpool_test.cpp

CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\malav\CLionProjects\ANN-master\ANN-master\testing\maxpool_test.cpp > CMakeFiles\maxpool_test.dir\testing\maxpool_test.cpp.i

CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\malav\CLionProjects\ANN-master\ANN-master\testing\maxpool_test.cpp -o CMakeFiles\maxpool_test.dir\testing\maxpool_test.cpp.s

# Object files for target maxpool_test
maxpool_test_OBJECTS = \
"CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.obj"

# External object files for target maxpool_test
maxpool_test_EXTERNAL_OBJECTS =

maxpool_test.exe: CMakeFiles/maxpool_test.dir/testing/maxpool_test.cpp.obj
maxpool_test.exe: CMakeFiles/maxpool_test.dir/build.make
maxpool_test.exe: CMakeFiles/maxpool_test.dir/linklibs.rsp
maxpool_test.exe: CMakeFiles/maxpool_test.dir/objects1.rsp
maxpool_test.exe: CMakeFiles/maxpool_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\malav\CLionProjects\ANN-master\ANN-master\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable maxpool_test.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\maxpool_test.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/maxpool_test.dir/build: maxpool_test.exe
.PHONY : CMakeFiles/maxpool_test.dir/build

CMakeFiles/maxpool_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\maxpool_test.dir\cmake_clean.cmake
.PHONY : CMakeFiles/maxpool_test.dir/clean

CMakeFiles/maxpool_test.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\malav\CLionProjects\ANN-master\ANN-master C:\Users\malav\CLionProjects\ANN-master\ANN-master C:\Users\malav\CLionProjects\ANN-master\ANN-master\cmake-build-debug C:\Users\malav\CLionProjects\ANN-master\ANN-master\cmake-build-debug C:\Users\malav\CLionProjects\ANN-master\ANN-master\cmake-build-debug\CMakeFiles\maxpool_test.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/maxpool_test.dir/depend

