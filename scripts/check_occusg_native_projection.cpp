// Link the unmodified official MapConverter.cpp. Six dispersed free cells must
// not substitute for a contiguous 0.3 m interval. An overlapping wall stays occupied.
#include "MapConverter.hh"
#include <iostream>
#include <stdexcept>
int main() {
  MapConverter converter(.05, 2, .3, 5);
  std::vector<voxel> voxels;
  for (int y=0; y<40; ++y) for (int x=0; x<60; ++x) {
    bool continuous = (x>=5 && x<=20 && y>=5 && y<=20);
    bool dispersed = (x>=35 && x<=50 && y>=25 && y<=35);
    bool wall = (x==21 && y>=5 && y<=20);
    for (int z=0; z<12; ++z) {
      if ((continuous && z<6) || (dispersed && z%2==0) || (wall && z<6))
        voxels.push_back({{(x+.5)*.05, (y+.5)*.05, (z+.5)*.05}, .025f, wall});
    }
  }
  converter.updateMap(voxels, {0., 3., 0., 2., 0., .6});
  const int continuous = converter.map.get(.525, .525);
  const int dispersed = converter.map.get(2.025, 1.525);
  const int wall = converter.map.get(1.075, .525);
  std::cout << "continuous=" << continuous << " dispersed=" << dispersed << " wall=" << wall << '\n';
  if (continuous != 0 || dispersed == 0 || wall <= 0)
    throw std::runtime_error("Official projection contract failed");
}
