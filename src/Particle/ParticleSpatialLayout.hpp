//
// Class ParticleSpatialLayout
//   Particle layout based on spatial decomposition.
//
//   This is a specialized version of ParticleLayout, which places particles
//   on processors based on their spatial location relative to a fixed grid.
//   In particular, this can maintain particles on processors based on a
//   specified FieldLayout or RegionLayout, so that particles are always on
//   the same node as the node containing the Field region to which they are
//   local.  This may also be used if there is no associated Field at all,
//   in which case a grid is selected based on an even distribution of
//   particles among processors.
//
//   After each 'time step' in a calculation, which is defined as a period
//   in which the particle positions may change enough to affect the global
//   layout, the user must call the 'update' routine, which will move
//   particles between processors, etc.  After the Nth call to update, a
//   load balancing routine will be called instead.  The user may set the
//   frequency of load balancing (N), or may supply a function to
//   determine if load balancing should be done or not.
//
#include <memory>
#include <numeric>
#include <vector>

#include "Utility/IpplTimings.h"

namespace ippl {

    template <typename T, unsigned Dim, class Mesh>
    ParticleSpatialLayout<T, Dim, Mesh>::ParticleSpatialLayout(FieldLayout<Dim>& fl,
                                                                              Mesh& mesh)
        : rlayout_m(fl, mesh) {}

    template <typename T, unsigned Dim, class Mesh>
    void ParticleSpatialLayout<T, Dim, Mesh>::updateLayout(FieldLayout<Dim>& fl,
                                                                          Mesh& mesh) {
        rlayout_m.changeDomain(fl, mesh);
    }

    template <typename T, unsigned Dim, class Mesh>
    template <size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialLayout<T, Dim, Mesh>::positionInRegion(
        const std::index_sequence<Idx...>&, const position_type& pos, const region_type& region) {
        return ((pos[Idx] >= region[Idx].min()) && ...) && ((pos[Idx] <= region[Idx].max()) && ...);
    };

    template <typename T, unsigned Dim, class Mesh>
    template <typename ParticleBunch>
    detail::size_type ParticleSpatialLayout<T, Dim, Mesh>::locateParticles(
        const ParticleBunch* pdata, typename ParticleBunch::locate_type& ranks,
        typename ParticleBunch::bool_type& invalid) {
        auto& positions                            = pdata.R.getView();
        typename RegionLayout_t::view_type Regions = rlayout_m.getdLocalRegions();

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename ParticleBunch::position_execution_space>;

        int myRank = Comm->rank();

        const auto is = std::make_index_sequence<Dim>{};

        size_type invalidCount = 0;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::locateParticles()",
            mdrange_type({0, 0}, {ranks.extent(0), Regions.extent(0)}),
            KOKKOS_LAMBDA(const size_t i, const size_type j, size_type& count) {
                bool xyz_bool = positionInRegion(is, positions(i), Regions(j));
                if (xyz_bool) {
                    ranks(i)   = j;
                    invalid(i) = (myRank != ranks(i));
                    count += invalid(i);
                }
            },
            Kokkos::Sum<size_type>(invalidCount));
        Kokkos::fence();

        return invalidCount;
    }
}  // namespace ippl
