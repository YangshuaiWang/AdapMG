/* ---------------------------------------------------------------------
 *
 * Adaptive multigrid: Read solution or estimated_error_per_cell of each step, output mesh information
 *
 * ---------------------------------------------------------------------
 *
 * Author: Yangshuai Wang, Shanghai Jiao Tong University, 2019
 */

// Hanging nodes are the ones that are constrained by a call to DoFTools::hanging_node_constraints.
// Hence, you could just call that function on an otherwise empty ConstraintMatrix (or AffineConstraints) object constraints
// and call constraints.print(std::cout).
// A different possibility is to just print the locations of all the degrees of freedom via a call to DoFTools::write_gnuplot_dof_support_point_info.


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

// #include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/lac/full_matrix.h>
// #include <deal.II/lac/precondition.h>
// #include <deal.II/lac/solver_cg.h>
// #include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream> // for reading and writing

#include <deal.II/fe/fe_q.h>
// grid_in.h:
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

#include <iostream> // add
#include<string> // add

using namespace dealii;
using namespace std; // add

template <int dim>
class Mesh
{
public:
  Mesh ();

  void run ();

private:
  void setup_system ();
  void refine_grid (const unsigned int cycle); // actually no cycle here, just step by step
  void output_results (const unsigned int cycle) const;

  Triangulation<dim> triangulation;

  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;
  ConstraintMatrix     constraints;
  
};

template <int dim>
Mesh<dim>::Mesh ()
  :
  fe (1),
  dof_handler (triangulation)
{}

template <int dim>
void Mesh<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);

  constraints.close ();
}

template <int dim>
void Mesh<dim>::refine_grid (const unsigned int cycle)
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  {
    ifstream est;
    est.open("estimator-" + std::to_string(cycle) + ".txt");
    for (unsigned int i=0; i<triangulation.n_active_cells(); ++i)
    {est >> estimated_error_per_cell[i]; }
    est.close();
  }

  float c = 0.1;
  // {
  //   ifstream coeff;
  //   coeff.open("coeff.txt");
  //   coeff >> c; 
  //   coeff.close();
  // }

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   c, 0.0);

  // ys: print the vector
  // std::cout << " estimated_error_per_cell: "
  //               << estimated_error_per_cell
  //               << std::endl;

  triangulation.execute_coarsening_and_refinement ();
}

template <int dim>
void Mesh<dim>::output_results (const unsigned int cycle) const
{
  // mesh information
  {
    GridOut grid_out;
    std::ofstream output ("grid-" + std::to_string(cycle) + ".msh");
    grid_out.write_msh (triangulation, output);
  }

  {
    GridOut grid_out;
    std::ofstream output ("grid-" + std::to_string(cycle) + ".eps");
    grid_out.write_eps (triangulation, output);
  }
  // constraints information
  {
    std::ofstream mycon ("constraints-" + std::to_string(cycle) + ".txt");
    constraints.print(mycon);
    cout << "Save constraints completed" << endl;
  }
}

template <int dim>
void Mesh<dim>::run ()
{ 
  unsigned int num;
  // // please change here only!
  {
    ifstream itnum;
    itnum.open("iteration.txt");
    itnum >> num; 
    itnum.close();
  }
  //unsigned int num = 2;
  // please change here only!

  for (unsigned int cycle=num; cycle<num+2; ++cycle) // 2 is fixed for one adaptive
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == num)
        // GridIn here
        // {
        //   GridGenerator::hyper_cube (triangulation);
        //   triangulation.refine_global (3);
        // }
        { GridIn<dim> grid_in;
          grid_in.attach_triangulation (triangulation);
          std::ifstream input_file ("grid-" + std::to_string(num) + ".msh");  // change here
          grid_in.read_msh (input_file);
        }
      else
      refine_grid (cycle);


      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

      setup_system ();

      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      output_results (cycle);
    }
}

int main ()
{

  try
    {
      //Mesh<2> AdapMG;
      Mesh<3> AdapMG;
      AdapMG.run ();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
