# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mxw/dev/medical-cnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mxw/dev/medical-cnn/build

# Include any dependencies generated for this target.
include CMakeFiles/medical-cnn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/medical-cnn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/medical-cnn.dir/flags.make

CMakeFiles/medical-cnn.dir/src/activation.cpp.o: CMakeFiles/medical-cnn.dir/flags.make
CMakeFiles/medical-cnn.dir/src/activation.cpp.o: ../src/activation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mxw/dev/medical-cnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/medical-cnn.dir/src/activation.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/medical-cnn.dir/src/activation.cpp.o -c /home/mxw/dev/medical-cnn/src/activation.cpp

CMakeFiles/medical-cnn.dir/src/activation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/medical-cnn.dir/src/activation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mxw/dev/medical-cnn/src/activation.cpp > CMakeFiles/medical-cnn.dir/src/activation.cpp.i

CMakeFiles/medical-cnn.dir/src/activation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/medical-cnn.dir/src/activation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mxw/dev/medical-cnn/src/activation.cpp -o CMakeFiles/medical-cnn.dir/src/activation.cpp.s

CMakeFiles/medical-cnn.dir/src/activation.cpp.o.requires:

.PHONY : CMakeFiles/medical-cnn.dir/src/activation.cpp.o.requires

CMakeFiles/medical-cnn.dir/src/activation.cpp.o.provides: CMakeFiles/medical-cnn.dir/src/activation.cpp.o.requires
	$(MAKE) -f CMakeFiles/medical-cnn.dir/build.make CMakeFiles/medical-cnn.dir/src/activation.cpp.o.provides.build
.PHONY : CMakeFiles/medical-cnn.dir/src/activation.cpp.o.provides

CMakeFiles/medical-cnn.dir/src/activation.cpp.o.provides.build: CMakeFiles/medical-cnn.dir/src/activation.cpp.o


CMakeFiles/medical-cnn.dir/src/f.cpp.o: CMakeFiles/medical-cnn.dir/flags.make
CMakeFiles/medical-cnn.dir/src/f.cpp.o: ../src/f.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mxw/dev/medical-cnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/medical-cnn.dir/src/f.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/medical-cnn.dir/src/f.cpp.o -c /home/mxw/dev/medical-cnn/src/f.cpp

CMakeFiles/medical-cnn.dir/src/f.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/medical-cnn.dir/src/f.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mxw/dev/medical-cnn/src/f.cpp > CMakeFiles/medical-cnn.dir/src/f.cpp.i

CMakeFiles/medical-cnn.dir/src/f.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/medical-cnn.dir/src/f.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mxw/dev/medical-cnn/src/f.cpp -o CMakeFiles/medical-cnn.dir/src/f.cpp.s

CMakeFiles/medical-cnn.dir/src/f.cpp.o.requires:

.PHONY : CMakeFiles/medical-cnn.dir/src/f.cpp.o.requires

CMakeFiles/medical-cnn.dir/src/f.cpp.o.provides: CMakeFiles/medical-cnn.dir/src/f.cpp.o.requires
	$(MAKE) -f CMakeFiles/medical-cnn.dir/build.make CMakeFiles/medical-cnn.dir/src/f.cpp.o.provides.build
.PHONY : CMakeFiles/medical-cnn.dir/src/f.cpp.o.provides

CMakeFiles/medical-cnn.dir/src/f.cpp.o.provides.build: CMakeFiles/medical-cnn.dir/src/f.cpp.o


CMakeFiles/medical-cnn.dir/src/main.cpp.o: CMakeFiles/medical-cnn.dir/flags.make
CMakeFiles/medical-cnn.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mxw/dev/medical-cnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/medical-cnn.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/medical-cnn.dir/src/main.cpp.o -c /home/mxw/dev/medical-cnn/src/main.cpp

CMakeFiles/medical-cnn.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/medical-cnn.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mxw/dev/medical-cnn/src/main.cpp > CMakeFiles/medical-cnn.dir/src/main.cpp.i

CMakeFiles/medical-cnn.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/medical-cnn.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mxw/dev/medical-cnn/src/main.cpp -o CMakeFiles/medical-cnn.dir/src/main.cpp.s

CMakeFiles/medical-cnn.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/medical-cnn.dir/src/main.cpp.o.requires

CMakeFiles/medical-cnn.dir/src/main.cpp.o.provides: CMakeFiles/medical-cnn.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/medical-cnn.dir/build.make CMakeFiles/medical-cnn.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/medical-cnn.dir/src/main.cpp.o.provides

CMakeFiles/medical-cnn.dir/src/main.cpp.o.provides.build: CMakeFiles/medical-cnn.dir/src/main.cpp.o


CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o: CMakeFiles/medical-cnn.dir/flags.make
CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o: ../src/nn_ops.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mxw/dev/medical-cnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o -c /home/mxw/dev/medical-cnn/src/nn_ops.cpp

CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mxw/dev/medical-cnn/src/nn_ops.cpp > CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.i

CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mxw/dev/medical-cnn/src/nn_ops.cpp -o CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.s

CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o.requires:

.PHONY : CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o.requires

CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o.provides: CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o.requires
	$(MAKE) -f CMakeFiles/medical-cnn.dir/build.make CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o.provides.build
.PHONY : CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o.provides

CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o.provides.build: CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o


# Object files for target medical-cnn
medical__cnn_OBJECTS = \
"CMakeFiles/medical-cnn.dir/src/activation.cpp.o" \
"CMakeFiles/medical-cnn.dir/src/f.cpp.o" \
"CMakeFiles/medical-cnn.dir/src/main.cpp.o" \
"CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o"

# External object files for target medical-cnn
medical__cnn_EXTERNAL_OBJECTS =

medical-cnn: CMakeFiles/medical-cnn.dir/src/activation.cpp.o
medical-cnn: CMakeFiles/medical-cnn.dir/src/f.cpp.o
medical-cnn: CMakeFiles/medical-cnn.dir/src/main.cpp.o
medical-cnn: CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o
medical-cnn: CMakeFiles/medical-cnn.dir/build.make
medical-cnn: /usr/lib/x86_64-linux-gnu/libarmadillo.so
medical-cnn: CMakeFiles/medical-cnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mxw/dev/medical-cnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable medical-cnn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/medical-cnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/medical-cnn.dir/build: medical-cnn

.PHONY : CMakeFiles/medical-cnn.dir/build

CMakeFiles/medical-cnn.dir/requires: CMakeFiles/medical-cnn.dir/src/activation.cpp.o.requires
CMakeFiles/medical-cnn.dir/requires: CMakeFiles/medical-cnn.dir/src/f.cpp.o.requires
CMakeFiles/medical-cnn.dir/requires: CMakeFiles/medical-cnn.dir/src/main.cpp.o.requires
CMakeFiles/medical-cnn.dir/requires: CMakeFiles/medical-cnn.dir/src/nn_ops.cpp.o.requires

.PHONY : CMakeFiles/medical-cnn.dir/requires

CMakeFiles/medical-cnn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/medical-cnn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/medical-cnn.dir/clean

CMakeFiles/medical-cnn.dir/depend:
	cd /home/mxw/dev/medical-cnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mxw/dev/medical-cnn /home/mxw/dev/medical-cnn /home/mxw/dev/medical-cnn/build /home/mxw/dev/medical-cnn/build /home/mxw/dev/medical-cnn/build/CMakeFiles/medical-cnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/medical-cnn.dir/depend

