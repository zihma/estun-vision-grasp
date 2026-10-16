#include "../src/visual_brick_target.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace estun_vision_grasp;

bool parse(const std::string& line, double now = 1000.1)
{
    VisualBrickTarget target;
    std::string reason;
    std::istringstream input(line);
    return parseVisualBrickTarget(input, now, target, reason);
}

bool parse_tape(const std::string& line, double now = 1000.1)
{
    VisualBrickTarget target;
    std::string reason;
    std::istringstream input(line);
    return parseVisualTapeTarget(input, now, target, reason);
}

bool parse_pringles(const std::string& line, double now = 1000.1)
{
    VisualBrickTarget target;
    std::string reason;
    std::istringstream input(line);
    return parseVisualPringlesTarget(input, now, target, reason);
}

int main(int argc, char** argv)
{
    const std::string good = "brick_v1 1000 base_link Foam_Brick 0.3 -0.1 0.03 0.9 1 0";
    assert(parse(good));
    assert(!parse(good, 1001.01));
    assert(!parse(good, 999.0));
    assert(!parse("INVALID NO_CONNECTED_TOP_FACE"));
    assert(!parse("1000 base_link 0.3 -0.1 0.03 0.9"));  // old XYZ target cannot pass
    assert(!parse("brick_v1 1000 camera Foam_Brick 0.3 -0.1 0.03 0.9 1 0"));
    assert(!parse("brick_v1 1000 base_link Mug 0.3 -0.1 0.03 0.9 1 0"));
    assert(!parse("brick_v1 1000 base_link Foam_Brick 0.3 -0.1 0.03 0.4 1 0"));
    assert(!parse("brick_v1 1000 base_link Foam_Brick 0.3 -0.1 0.03 0.9 0 0"));
    assert(!parse("brick_v1 1000 base_link Foam_Brick nan -0.1 0.03 0.9 1 0"));
    assert(!parse(good + " unexpected"));
    const std::string tape = "tape_v1 1000 base_link Scotch_Duct_Tape 0.28 -0.04 0.02 0.99 -0.37 -0.929031754";
    assert(parse_tape(tape));
    assert(!parse(tape));
    assert(!parse_tape(good));
    assert(!parse_tape(tape, 1001.1));
    assert(!parse_tape("tape_v1 1000 base_link Foam_Brick 0.28 -0.04 0.02 0.99 -0.37 -0.929031754"));
    const std::string pringles = "pringles_v1 1000 base_link Pringles 0.28 -0.04 0.04 0.81 -0.853653 -0.520842";
    assert(parse_pringles(pringles));
    assert(!parse(pringles));
    assert(!parse_tape(pringles));
    assert(!parse_pringles(good));
    VisualBrickTarget initial{1000, {0.3, -0.1, 0.03}, 0.9, {1, 0}};
    auto latest = initial;
    latest.short_axis = {-1, 0};
    assert(sameVisualBrickTarget(initial, latest));
    latest.short_axis = {0, 1};
    assert(!sameVisualBrickTarget(initial, latest));
    latest = initial;
    latest.xyz[0] += 0.02;
    assert(!sameVisualBrickTarget(initial, latest));

    // Verify local +45 degree closing axis transforms into the desired
    // base XY axis for both equivalent downward target orientations.
    for (int degrees = 0; degrees < 360; degrees += 10)
    {
        const double a = degrees * 3.14159265358979323846 / 180.0;
        initial.short_axis = {std::cos(a), std::sin(a)};
        for (int flip = 0; flip < 2; ++flip)
        {
            const double yaw = visualToolYaw(initial) + flip * 3.14159265358979323846;
            const double x = std::cos(yaw) * std::cos(kClosingOffsetRad) +
                             std::sin(yaw) * std::sin(kClosingOffsetRad);
            const double y = std::sin(yaw) * std::cos(kClosingOffsetRad) -
                             std::cos(yaw) * std::sin(kClosingOffsetRad);
            assert(std::abs(std::abs(x * initial.short_axis[0] + y * initial.short_axis[1]) - 1) < 1e-12);
        }
    }
    if (argc > 1)
    {
        std::ifstream fixture(argv[1]);
        assert(fixture.is_open());
        std::string line;
        int count = 0;
        while (std::getline(fixture, line))
        {
            assert(parse(line));
            ++count;
        }
        assert(count > 0);
        std::cout << "Validated " << count << " Python-generated real-snapshot targets\n";
    }
    std::cout << "Visual target parsing, freshness, drift and +45 degree yaw checks passed\n";
}
