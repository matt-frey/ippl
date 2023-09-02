//
// Class ParticleLayout
//   Base class for all particle layout classes.
//
//   This class is used as the generic base class for all classes
//   which maintain the information on where all the particles are located
//   on a parallel machine.  It is responsible for performing particle
//   load balancing.
//
//   If more general layout information is needed, such as the global -> local
//   mapping for each particle, then derived classes must provide this info.
//
//   When particles are created or destroyed, this class is also responsible
//   for determining where particles are to be created, gathering this
//   information, and recalculating the global indices of all the particles.
//   For consistency, creation and destruction requests are cached, and then
//   performed all in one step when the update routine is called.
//
//   Derived classes must provide the following:
//     1) Specific version of update and loadBalance.  These are not virtual,
//        as this class is used as a template parameter (instead of being
//        assigned to a base class pointer).
//     2) Internal storage to maintain their specific layout mechanism
//     3) the definition of a class pair_iterator, and a function
//        void getPairlist(int, pair_iterator&, pair_iterator&) to get a
//        begin/end iterator pair to access the local neighbor of the Nth
//        local atom.  This is not a virtual function, it is a requirement of
//        the templated class for use in other parts of the code.
//

#ifndef IPPL_PARTICLE_LAYOUT_H
#define IPPL_PARTICLE_LAYOUT_H

#include "Types/IpplTypes.h"
#include "ParticleAttrib.h"

namespace ippl {
    /*!
     * ParticleLayout class definition.
     */
    template <typename T, unsigned Dim>
    class ParticleLayout {
    public:
        using position_type = Vector<T, Dim>;
        using position_execution_space = Kokkos::DefaultExecutionSpace;
        using position_memory_space = position_execution_space::memory_space;
        using hash_type   = detail::hash_type<position_memory_space>;
        using locate_type = typename detail::ViewType<int, 1, position_memory_space>::view_type;
        using bool_type   = typename detail::ViewType<bool, 1, position_memory_space>::view_type;

        using size_type = detail::size_type;

    public:
        // constructor: this one also takes a Mesh
        ParticleLayout() = default;

        ~ParticleLayout() = default;

        template <class ParticleContainer>
        void update(ParticleContainer* pc);

        /*!
         * For each particle in the bunch, determine the rank on which it should
         * be stored based on its location
         * @param pdata the particle bunch
         * @param ranks the integer view in which to store the destination ranks
         * @param invalid the boolean view in which to store whether each particle
         * is invalidated, i.e. needs to be sent to another rank
         * @return The total number of invalidated particles
         */
        virtual size_type locateParticles(const ParticleAttrib<position_type>* pos,
                                  locate_type& ranks,
                                  bool_type& invalid) = 0;

        /*!
         * @param rank we sent to
         * @param ranks a container specifying where a particle at the i-th index should go.
         * @param hash a mapping to fill the send buffer contiguously
         */
        template <class ParticleContainer>
        void fillHash(int rank, const locate_type& ranks,
                      hash_type & hash);

        /*!
         * @param rank we sent to
         * @param ranks a container specifying where a particle at the i-th index should go.
         */
        template <class ParticleContainer>
        size_t numberOfSends(int rank, const locate_type& ranks);
    };
}  // namespace ippl

#include "Particle/ParticleLayout.hpp"

#endif
