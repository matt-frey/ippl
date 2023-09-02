#include <memory>
#include <numeric>
#include <vector>

#include "Utility/IpplTimings.h"

namespace ippl {

    template <typename T, unsigned Dim>
    template <class ParticleContainer>
    void ParticleLayout<T, Dim>::update(ParticleContainer* pc) {

//         static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
//         IpplTimings::startTimer(ParticleBCTimer);
//         this->applyBC(pc->R, rlayout_m.getDomain());
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

        auto* pos = pc->getPositionAttribute();

        size_type invalidCount = locateParticles(pos, ranks, invalid);
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
            nSends[rank] = numberOfSends<ParticleContainer>(rank, ranks);
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
                fillHash<ParticleContainer>(rank, ranks, hash);

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

//     template <class ParticleContainer>
//     ParticleLayout::size_type ParticleLayout::locateParticles(
//         const ParticleContainer* /*pc*/,
//         locate_type& /*ranks*/,
//         bool_type& /*invalid*/) {
// //         auto& positions                            = pc.getPositionAttribute().getView();
// //         typename RegionLayout_t::view_type Regions = rlayout_m.getdLocalRegions();
// //
// //         using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
// //
// //         int myRank = Comm->rank();
// //
// //         const auto is = std::make_index_sequence<Dim>{};
// //
// //         size_type invalidCount = 0;
// //         Kokkos::parallel_reduce(
// //             "ParticleLayout::locateParticles()",
// //             mdrange_type({0, 0}, {ranks.extent(0), Regions.extent(0)}),
// //             KOKKOS_LAMBDA(const size_t i, const size_type j, size_type& count) {
// //                 bool xyz_bool = positionInRegion(is, positions(i), Regions(j));
// //                 if (xyz_bool) {
// //                     ranks(i)   = j;
// //                     invalid(i) = (myRank != ranks(i));
// //                     count += invalid(i);
// //                 }
// //             },
// //             Kokkos::Sum<size_type>(invalidCount));
// //         Kokkos::fence();
// //
// //         return invalidCount;
//         std::cout << "Not implemented." << std::endl;
//         return 0;
//     }


    template <typename T, unsigned Dim>
    template <class ParticleContainer>
    void ParticleLayout<T, Dim>::fillHash(int rank,
                                  const locate_type& ranks,
                                  hash_type& hash) {
        /* Compute the prefix sum and fill the hash
         */
        using execution_space = position_execution_space;
        using policy_type = Kokkos::RangePolicy<execution_space>;
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


    template <typename T, unsigned Dim>
    template <class ParticleContainer>
    size_t ParticleLayout<T, Dim>::numberOfSends(
        int rank, const locate_type& ranks) {
        size_t nSends     = 0;
        using execution_space = position_execution_space;
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_reduce(
            "ParticleLayout::numberOfSends()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, size_t& num) { num += size_t(rank == ranks(i)); },
            nSends);
        Kokkos::fence();
        return nSends;
    }
}  // namespace ippl
