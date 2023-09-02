
namespace ippl {

    template <typename... IP>
    ParticleBase<IP...>::ParticleBase()
        : layout_m(nullptr)
        , localNum_m(0)
        , nextID_m(Comm->rank())
        , numNodes_m(Comm->size()) {
        if constexpr (EnableIDs) {
            addAttribute(ID);
        }
    }

    template <typename... IP>
    ParticleBase<IP...>::ParticleBase(std::shared_ptr<ParticleLayout> layout)
        : ParticleBase()
    {
        this->initialize(layout);
    }

    template <typename... IP>
    template <typename MemorySpace>
    void ParticleBase<IP...>::addAttribute(detail::ParticleAttribBase<MemorySpace>& pa) {
        attributes_m.template get<MemorySpace>().push_back(&pa);
        pa.setParticleCount(localNum_m);
    }

    template <typename... IP>
    void ParticleBase<IP...>::addPositionAttribute(detail::ParticleAttribBase<position_memory_space>& pa) {
        attributes_m.template get<position_memory_space>().push_back(&pa);
        positionIndex_m = attributes_m.template get<position_memory_space>().size();
    }

    template <typename... IP>
    void ParticleBase<IP...>::initialize(std::shared_ptr<ParticleLayout> layout) {
        layout_m = std::move(layout);
    }

    template <typename... IP>
    void ParticleBase<IP...>::create(size_type nLocal) {
        PAssert(layout_m != nullptr);

        forAllAttributes([&]<typename Attribute>(Attribute& attribute) {
            attribute->create(nLocal);
        });

        if constexpr (EnableIDs) {
            // set the unique ID value for these new particles
            using policy_type =
                Kokkos::RangePolicy<size_type, typename particle_index_type::execution_space>;
            auto pIDs     = ID.getView();
            auto nextID   = this->nextID_m;
            auto numNodes = this->numNodes_m;
            Kokkos::parallel_for(
                "ParticleBase<...>::create(size_t)", policy_type(localNum_m, nLocal),
                KOKKOS_LAMBDA(const std::int64_t i) { pIDs(i) = nextID + numNodes * i; });
            // nextID_m += numNodes_m * (nLocal - localNum_m);
            nextID_m += numNodes_m * nLocal;
        }

        // remember that we're creating these new particles
        localNum_m += nLocal;
    }

    template <typename... IP>
    void ParticleBase<IP...>::createWithID(index_type id) {
        PAssert(layout_m != nullptr);

        // temporary change
        index_type tmpNextID = nextID_m;
        nextID_m             = id;
        numNodes_m           = 0;

        create(1);

        nextID_m   = tmpNextID;
        numNodes_m = Comm->getNodes();
    }

    template <typename... IP>
    void ParticleBase<IP...>::globalCreate(size_type nTotal) {
        PAssert(layout_m != nullptr);

        // Compute the number of particles local to each processor
        size_type nLocal = nTotal / numNodes_m;

        const size_t rank = Comm->myNode();

        size_type rest = nTotal - nLocal * rank;
        if (rank < rest) {
            ++nLocal;
        }

        create(nLocal);
    }

    template <typename... IP>
    template <typename... Properties>
    void ParticleBase<IP...>::destroy(const Kokkos::View<bool*, Properties...>& invalid,
                                               const size_type destroyNum) {
        PAssert(destroyNum <= localNum_m);

        // If there aren't any particles to delete, do nothing
        if (destroyNum == 0) {
            return;
        }

        // If we're deleting all the particles, there's no point in doing
        // anything because the valid region will be empty; we only need to
        // update the particle count
        if (destroyNum == localNum_m) {
            localNum_m = 0;
            return;
        }

        using view_type       = Kokkos::View<bool*, Properties...>;
        using memory_space    = typename view_type::memory_space;
        using execution_space = typename view_type::execution_space;
        using policy_type     = Kokkos::RangePolicy<execution_space>;
        auto& locDeleteIndex  = deleteIndex_m.get<memory_space>();
        auto& locKeepIndex    = keepIndex_m.get<memory_space>();

        // Resize buffers, if necessary
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            if (attributes_m.template get<memory_space>().size() > 0) {
                int overalloc = Comm->getDefaultOverallocation();
                auto& del     = deleteIndex_m.get<memory_space>();
                auto& keep    = keepIndex_m.get<memory_space>();
                if (del.size() < destroyNum) {
                    Kokkos::realloc(del, destroyNum * overalloc);
                    Kokkos::realloc(keep, destroyNum * overalloc);
                }
            }
        });

        // Reset index buffer
        Kokkos::deep_copy(locDeleteIndex, -1);

        // Find the indices of the invalid particles in the valid region
        Kokkos::parallel_scan(
            "Scan in ParticleBase::destroy()", policy_type(0, localNum_m - destroyNum),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final && invalid(i)) {
                    locDeleteIndex(idx) = i;
                }
                if (invalid(i)) {
                    idx += 1;
                }
            });
        Kokkos::fence();

        // Determine the total number of invalid particles in the valid region
        size_type maxDeleteIndex = 0;
        Kokkos::parallel_reduce(
            "Reduce in ParticleBase::destroy()", policy_type(0, destroyNum),
            KOKKOS_LAMBDA(const size_t i, size_t& maxIdx) {
                if (locDeleteIndex(i) >= 0 && i > maxIdx) {
                    maxIdx = i;
                }
            },
            Kokkos::Max<size_type>(maxDeleteIndex));

        // Find the indices of the valid particles in the invalid region
        Kokkos::parallel_scan(
            "Second scan in ParticleBase::destroy()",
            Kokkos::RangePolicy<size_type, execution_space>(localNum_m - destroyNum, localNum_m),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final && !invalid(i)) {
                    locKeepIndex(idx) = i;
                }
                if (!invalid(i)) {
                    idx += 1;
                }
            });

        Kokkos::fence();

        localNum_m -= destroyNum;

        auto filter = [&]<typename MemorySpace>() {
            return attributes_m.template get<MemorySpace>().size() > 0;
        };
        deleteIndex_m.copyToOtherSpaces<memory_space>(filter);
        keepIndex_m.copyToOtherSpaces<memory_space>(filter);

        // Partition the attributes into valid and invalid regions
        // NOTE: The vector elements are pointers, but we want to extract
        // the memory space from the class type, so we explicitly
        // make the lambda argument a pointer to the template parameter
        forAllAttributes([&]<typename Attribute>(Attribute*& attribute) {
            using att_memory_space = typename Attribute::memory_space;
            auto& del              = deleteIndex_m.get<att_memory_space>();
            auto& keep             = keepIndex_m.get<att_memory_space>();
            attribute->destroy(del, keep, maxDeleteIndex + 1);
        });
    }

    template <typename... IP>
    template <typename HashType>
    void ParticleBase<IP...>::sendToRank(int rank, int tag, int sendNum,
                                                  std::vector<MPI_Request>& requests,
                                                  const HashType& hash) {
        size_type nSends = hash.size();
        requests.resize(requests.size() + 1);

        auto hashes = hash_container_type(hash, [&]<typename MemorySpace>() {
            return attributes_m.template get<MemorySpace>().size() > 0;
        });
        pack(hashes);
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nSends);
            if (bufSize == 0) {
                return;
            }

            auto buf = Comm->getBuffer<MemorySpace>(IPPL_PARTICLE_SEND + sendNum, bufSize);

            Comm->isend(rank, tag++, *this, *buf, requests.back(), nSends);
            buf->resetWritePos();
        });
    }

    template <typename... IP>
    void ParticleBase<IP...>::recvFromRank(int rank, int tag, int recvNum,
                                                    size_type nRecvs) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nRecvs);
            if (bufSize == 0) {
                return;
            }

            auto buf = Comm->getBuffer<MemorySpace>(IPPL_PARTICLE_RECV + recvNum, bufSize);

            Comm->recv(rank, tag++, *this, *buf, bufSize, nRecvs);
            buf->resetReadPos();
        });
        unpack(nRecvs);
    }

    template <typename... IP>
    template <typename Archive>
    void ParticleBase<IP...>::serialize(Archive& ar, size_type nsends) {
        using memory_space = typename Archive::buffer_type::memory_space;
        forAllAttributes<memory_space>([&]<typename Attribute>(Attribute& att) {
            att->serialize(ar, nsends);
        });
    }

    template <typename... IP>
    template <typename Archive>
    void ParticleBase<IP...>::deserialize(Archive& ar, size_type nrecvs) {
        using memory_space = typename Archive::buffer_type::memory_space;
        forAllAttributes<memory_space>([&]<typename Attribute>(Attribute& att) {
            att->deserialize(ar, nrecvs);
        });
    }

    template <typename... IP>
    template <typename MemorySpace>
    detail::size_type ParticleBase<IP...>::packedSize(const size_type count) const {
        size_type total = 0;
        forAllAttributes<MemorySpace>([&]<typename Attribute>(const Attribute& att) {
            total += att->packedSize(count);
        });
        return total;
    }

    template <typename... IP>
    void ParticleBase<IP...>::update() { layout_m->update(this); };

    template <typename... IP>
    void ParticleBase<IP...>::pack(const hash_container_type& hash) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            auto& att = attributes_m.template get<MemorySpace>();
            for (unsigned j = 0; j < att.size(); j++) {
                att[j]->pack(hash.template get<MemorySpace>());
            }
        });
    }

    template <typename... IP>
    void ParticleBase<IP...>::unpack(size_type nrecvs) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            auto& att = attributes_m.template get<MemorySpace>();
            for (unsigned j = 0; j < att.size(); j++) {
                att[j]->unpack(nrecvs);
            }
        });
        localNum_m += nrecvs;
    }
}  // namespace ippl
