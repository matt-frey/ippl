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
#include <memory>
#include <numeric>
#include <vector>

#include "Utility/IpplTimings.h"

namespace ippl {

    template <typename T, unsigned Dim, class Mesh, class ExecutionSpace>
    ParticleLayout<T, Dim, Mesh, ExecutionSpace>::ParticleLayout(FieldLayout<Dim>& fl,
                                                                              Mesh& mesh)
        : rlayout_m(fl, mesh) {}

    template <typename T, unsigned Dim, class Mesh, class ExecutionSpace>
    void ParticleLayout<T, Dim, Mesh, ExecutionSpace>::updateLayout(FieldLayout<Dim>& fl,
                                                                          Mesh& mesh) {
        rlayout_m.changeDomain(fl, mesh);
    }

    template <typename T, unsigned Dim, class Mesh, class ExecutionSpace>
    template <class ParticleContainer>
    void ParticleLayout<T, Dim, Mesh, ExecutionSpace>::update(ParticleContainer* pc) {

        using locate_type = typename ParticleContainer::locate_type;
        using bool_type = typename ParticleContainer::bool_type;
        using hash_type = typename ParticleContainer::hash_type;


//         static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
//         IpplTimings::startTimer(ParticleBCTimer);
//         this->applyBC(pdata.R, rlayout_m.getDomain());
//         IpplTimings::stopTimer(ParticleBCTimer);

        static IpplTimings::TimerRef ParticleUpdateTimer = IpplTimings::getTimer("updateParticle");
        IpplTimings::startTimer(ParticleUpdateTimer);
        int nRanks = Comm->size();

        if (nRanks < 2) {
            return;
        }

        /* particle MPI exchange:
         *   1. figure out which particles need to go where
         *   2. fill send buffer and send particles
         *   3. delete invalidated particles
         *   4. receive particles
         */

        static IpplTimings::TimerRef locateTimer = IpplTimings::getTimer("locateParticles");
        IpplTimings::startTimer(locateTimer);
        size_type localnum = pc->getLocalNum();

        // 1st step

        /* the values specify the rank where
         * the particle with that index should go
         */
        locate_type ranks("MPI ranks", localnum);

        /* 0 --> particle valid
         * 1 --> particle invalid
         */
        bool_type invalid("invalid", localnum);

        size_type invalidCount = locateParticles(pc, ranks, invalid);
        IpplTimings::stopTimer(locateTimer);

        // 2nd step

        // figure out how many receives
        static IpplTimings::TimerRef preprocTimer = IpplTimings::getTimer("sendPreprocess");
        IpplTimings::startTimer(preprocTimer);
        MPI_Win win;
        std::vector<size_type> nRecvs(nRanks, 0);
        MPI_Win_create(nRecvs.data(), nRanks * sizeof(size_type), sizeof(size_type), MPI_INFO_NULL,
                       Comm->getCommunicator(), &win);

        std::vector<size_type> nSends(nRanks, 0);

        MPI_Win_fence(0, win);

        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == Comm->rank()) {
                // we do not need to send to ourselves
                continue;
            }
            nSends[rank] = numberOfSends(rank, ranks);
            MPI_Put(nSends.data() + rank, 1, MPI_LONG_LONG_INT, rank, Comm->rank(), 1,
                    MPI_LONG_LONG_INT, win);
        }
        MPI_Win_fence(0, win);
        MPI_Win_free(&win);
        IpplTimings::stopTimer(preprocTimer);

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        IpplTimings::startTimer(sendTimer);
        // send
        std::vector<MPI_Request> requests(0);

        int tag = Comm->next_tag(P_SPATIAL_LAYOUT_TAG, P_LAYOUT_CYCLE);

        int sends = 0;
        for (int rank = 0; rank < nRanks; ++rank) {
            if (nSends[rank] > 0) {
                hash_type hash("hash", nSends[rank]);
                fillHash(rank, ranks, hash);

                pc->sendToRank(rank, tag, sends++, requests, hash);
            }
        }
        IpplTimings::stopTimer(sendTimer);

        // 3rd step
        static IpplTimings::TimerRef destroyTimer = IpplTimings::getTimer("particleDestroy");
        IpplTimings::startTimer(destroyTimer);

        pc->destroy(invalid, invalidCount);
        Kokkos::fence();

        IpplTimings::stopTimer(destroyTimer);
        static IpplTimings::TimerRef recvTimer = IpplTimings::getTimer("particleRecv");
        IpplTimings::startTimer(recvTimer);
        // 4th step
        int recvs = 0;
        for (int rank = 0; rank < nRanks; ++rank) {
            if (nRecvs[rank] > 0) {
                pc->recvFromRank(rank, tag, recvs++, nRecvs[rank]);
            }
        }
        IpplTimings::stopTimer(recvTimer);

        IpplTimings::startTimer(sendTimer);

        if (requests.size() > 0) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        IpplTimings::stopTimer(sendTimer);

        IpplTimings::stopTimer(ParticleUpdateTimer);
    }

    template <typename T, unsigned Dim, class Mesh, class ExecutionSpace>
    template <size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleLayout<T, Dim, Mesh, ExecutionSpace>::positionInRegion(
        const std::index_sequence<Idx...>&, const Vector<T, Dim>& pos, const region_type& region) {
        return ((pos[Idx] >= region[Idx].min()) && ...) && ((pos[Idx] <= region[Idx].max()) && ...);
    };

    template <typename T, unsigned Dim, class Mesh, class ExecutionSpace>
    template <class ParticleContainer>
    detail::size_type ParticleLayout<T, Dim, Mesh, ExecutionSpace>::locateParticles(
        const ParticleContainer* pc, typename ParticleContainer::locate_type& ranks,
        typename ParticleContainer::bool_type& invalid) const {
        auto& positions                            = pc->R_p->getView();
        typename RegionLayout_t::view_type Regions = rlayout_m.getdLocalRegions();

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>;

        int myRank = Comm->rank();

        const auto is = std::make_index_sequence<Dim>{};

        size_type invalidCount = 0;
        Kokkos::parallel_reduce(
            "ParticleLayout::locateParticles()",
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

    template <typename T, unsigned Dim, class Mesh, class ExecutionSpace>
    template <class ParticleContainer>
    void ParticleLayout<T, Dim, Mesh, ExecutionSpace>::fillHash(int rank,
                                                                const typename ParticleContainer::locate_type& ranks,
                                                                      typename ParticleContainer::hash_type& hash) {
        /* Compute the prefix sum and fill the hash
         */
        using policy_type = Kokkos::RangePolicy<ExecutionSpace>;
        Kokkos::parallel_scan(
            "ParticleLayout::fillHash()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final) {
                    if (rank == ranks(i)) {
                        hash(idx) = i;
                    }
                }

                if (rank == ranks(i)) {
                    idx += 1;
                }
            });
        Kokkos::fence();
    }

    template <typename T, unsigned Dim, class Mesh, class ExecutionSpace>
    template <class ParticleContainer>
    size_t ParticleLayout<T, Dim, Mesh, ExecutionSpace>::numberOfSends(
        int rank, const typename ParticleContainer::locate_type& ranks) {
        size_t nSends     = 0;
        using policy_type = Kokkos::RangePolicy<ExecutionSpace>;
        Kokkos::parallel_reduce(
            "ParticleLayout::numberOfSends()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, size_t& num) { num += size_t(rank == ranks(i)); },
            nSends);
        Kokkos::fence();
        return nSends;
    }
}  // namespace ippl
