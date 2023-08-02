file(REMOVE_RECURSE
  "libglog.a"
  "libglog.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/glog.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
