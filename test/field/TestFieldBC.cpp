#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>
#include <fstream>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    int pt = 8;
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned,allParallel);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};

    using Mesh_t = ippl::UniformCartesian<double, 3>;

    Mesh_t mesh(owned, hx, origin);


    typedef ippl::Field<double, dim> field_type;

    typedef ippl::BConds<double, dim> bc_type; 

    bc_type bcField;

    //X direction periodic BC
    for (unsigned int i=0; i < 2; ++i) {
        bcField[i] = std::make_shared<ippl::PeriodicFace<double, dim>>(i);
    }
    ////////Lower Y face 
    bcField[2] = std::make_shared<ippl::NoBcFace<double, dim>>(2);
    ////Higher Y face
    bcField[3] = std::make_shared<ippl::ConstantFace<double, dim>>(3, 7.0);
    ////Lower Z face
    bcField[4] = std::make_shared<ippl::ZeroFace<double, dim>>(4);
    ////Higher Z face
    bcField[5] = std::make_shared<ippl::ExtrapolateFace<double, dim>>(5, 0.0, 1.0);

    //std::cout << bcField << std::endl;
    std::cout << layout << std::endl;

    field_type field(mesh, layout);

    field = 1.0;

    field = field * 10.0;

    bcField.findBCNeighbors(field);

    unsigned int niter = 5;


    for (unsigned int i=0; i < niter; ++i) {
        bcField.apply(field);
    }

    int nRanks = Ippl::Comm->size();
    for (int rank = 0; rank < nRanks; ++rank) {
        if (rank == Ippl::Comm->rank()) {
            std::ofstream out("field_allBC_" + 
                             std::to_string(rank) + 
                             ".dat", std::ios::out);
            field.write(out);
            out.close();
        }
        Ippl::Comm->barrier();
    }

    return 0;
}
