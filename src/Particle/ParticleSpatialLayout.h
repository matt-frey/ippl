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
#ifndef IPPL_PARTICLE_SPATIAL_LAYOUT_H
#define IPPL_PARTICLE_SPATIAL_LAYOUT_H

#include "Types/IpplTypes.h"

#include "FieldLayout/FieldLayout.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleLayout.h"
#include "Region/RegionLayout.h"

namespace ippl {
    /*!
     * ParticleSpatialLayout class definition.
     * @tparam T value type
     * @tparam Dim dimension
     * @tparam Mesh type
     */
    template <typename T, unsigned Dim, class Mesh>
    class ParticleSpatialLayout : public ParticleLayout {
    public:
//         using particle_position_type   = ParticleAttrib<T, PositionProperties...>;
//         using position_memory_space    = typename particle_position_type::memory_space;
//         using position_execution_space = typename particle_position_type::execution_space;

//         using hash_type   = detail::hash_type<position_memory_space>;
//         using locate_type = typename detail::ViewType<int, 1, position_memory_space>::view_type;
//         using bool_type   = typename detail::ViewType<bool, 1, position_memory_space>::view_type;

        using position_type = T;
        using RegionLayout_t =
            typename detail::RegionLayout<T, Dim, Mesh, Kokkos::DefaultExecutionSpace::memory_space>::uniform_type;

        using size_type = detail::size_type;

    public:
        // constructor: this one also takes a Mesh
        ParticleSpatialLayout(FieldLayout<Dim>&, Mesh&);

        ParticleSpatialLayout()
            : ParticleLayout() {}

        ~ParticleSpatialLayout() = default;

        void updateLayout(FieldLayout<Dim>&, Mesh&);

    protected:
        //! The RegionLayout which determines where our particles go.
        RegionLayout_t rlayout_m;

        using region_type = typename RegionLayout_t::view_type::value_type;

        template <size_t... Idx>
        KOKKOS_INLINE_FUNCTION constexpr static bool positionInRegion(
            const std::index_sequence<Idx...>&, const position_type& pos, const region_type& region);

    public:
        /*!
         * For each particle in the bunch, determine the rank on which it should
         * be stored based on its location
         * @tparam ParticleBunch the bunch type
         * @param pdata the particle bunch
         * @param ranks the integer view in which to store the destination ranks
         * @param invalid the boolean view in which to store whether each particle
         * is invalidated, i.e. needs to be sent to another rank
         * @return The total number of invalidated particles
         */
        template <typename ParticleBunch>
        size_type locateParticles(const ParticleBunch* pdata, typename ParticleBunch::locate_type& ranks,
                                  typename ParticleBunch::bool_type& invalid);

    };
}  // namespace ippl

#include "Particle/ParticleSpatialLayout.hpp"

#endif
