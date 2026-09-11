// Data adapter only: no projection, DUDE, region filtering, or GT access.
// Output is the exact writeData payload used by octomap_msgs::fullMapToMsg.
#include <octomap/OcTree.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

int main(int argc, char **argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: adapter resolution records.bin output.data");
    double resolution = std::stod(argv[1]);
    octomap::OcTree tree(resolution);
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open records");
    float record[4];
    std::uint64_t count = 0;
    while (input.read(reinterpret_cast<char *>(record), sizeof(record))) {
      for (int i = 0; i < 4; ++i)
        if (!std::isfinite(record[i])) throw std::runtime_error("nonfinite record");
      if (record[3] != 1 && record[3] != 2) throw std::runtime_error("non-binary known state");
      tree.setNodeValue(record[0], record[1], record[2], record[3] == 2 ? 3.5f : -2.0f, true);
      ++count;
    }
    if (input.gcount() != 0) throw std::runtime_error("truncated records");
    if (!count) throw std::runtime_error("no known voxels");
    tree.updateInnerOccupancy();
    tree.prune();
    std::ofstream output(argv[3], std::ios::binary);
    tree.writeData(output);
    if (!output) throw std::runtime_error("failed to save OctoMap payload");
    std::cout << "known_voxels=" << count << " octree_leaves=" << tree.getNumLeafNodes() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
