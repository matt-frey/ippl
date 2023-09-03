//
// Class ParticleLayout
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
#ifndef IPPL_PARTICLE_LAYOUT_H
#define IPPL_PARTICLE_LAYOUT_H

#include "Types/IpplTypes.h"

#include "FieldLayout/FieldLayout.h"
#include "Particle/ParticleBase.h"
#include "Region/RegionLayout.h"

namespace ippl {
    /*!
     * ParticleLayout class definition.
     * @tparam T value type
     * @tparam Dim dimension
     * @tparam Mesh type
     */
    template <typename T, unsigned Dim, class Mesh, class ExecutionSpace>
    class ParticleLayout {
    public:
        using RegionLayout_t =
            typename detail::RegionLayout<T, Dim, Mesh, typename ExecutionSpace::memory_space>::uniform_type;

        using size_type = detail::size_type;

    public:
        // constructor: this one also takes a Mesh
        ParticleLayout(FieldLayout<Dim>&, Mesh&);

        ParticleLayout() = default;

        ~ParticleLayout() = default;

        void updateLayout(FieldLayout<Dim>&, Mesh&);

        template <class ParticleContainer>
        void update(ParticleContainer* pc);

        const RegionLayout_t& getRegionLayout() const { return rlayout_m; }

    protected:
        //! The RegionLayout which determines where our particles go.
        RegionLayout_t rlayout_m;

        using region_type = typename RegionLayout_t::view_type::value_type;

        template <size_t... Idx>
        KOKKOS_INLINE_FUNCTION constexpr static bool positionInRegion(
            const std::index_sequence<Idx...>&, const Vector<T, Dim>& pos, const region_type& region);

    public:
        /*!
         * For each particle in the bunch, determine the rank on which it should
         * be stored based on its location
         * @tparam ParticleContainer the particle container type
         * @param pdata the particle bunch
         * @param ranks the integer view in which to store the destination ranks
         * @param invalid the boolean view in which to store whether each particle
         * is invalidated, i.e. needs to be sent to another rank
         * @return The total number of invalidated particles
         */
        template <class ParticleContainer>
        size_type locateParticles(const ParticleContainer* pc,
                                  typename ParticleContainer::locate_type& ranks,
                                  typename ParticleContainer::bool_type& invalid) const;

        /*!
         * @param rank we sent to
         * @param ranks a container specifying where a particle at the i-th index should go.
         * @param hash a mapping to fill the send buffer contiguously
         */
        template <class ParticleContainer>
        void fillHash(int rank,
                      const typename ParticleContainer::locate_type& ranks,
                      typename ParticleContainer::hash_type& hash);

        /*!
         * @param rank we sent to
         * @param ranks a container specifying where a particle at the i-th index should go.
         */
        template <class ParticleContainer>
        size_t numberOfSends(int rank, const typename ParticleContainer::locate_type& ranks);
    };
}  // namespace ippl

#include "Particle/ParticleLayout.hpp"

#endif
