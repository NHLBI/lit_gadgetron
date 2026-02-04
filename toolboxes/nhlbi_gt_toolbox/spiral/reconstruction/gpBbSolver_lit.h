#pragma once

#include <gpSolver.h>
#include <real_utilities.h>
#include <complext.h>
#include <cgPreconditioner.h>
#include <vector>
#include <iostream>
#include <util_functions.h>
namespace Gadgetron
{

    /* Using adaptive step size from Zhou et al, 2006, Computational Optimization and Applications,
     * DOI: 10.1007/s10589-006-6446-0
     */

    template <class ARRAY_TYPE>
    class gpBbSolver_lit : public gpSolver<ARRAY_TYPE>
    {
    protected:
        typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
        typedef typename realType<ELEMENT_TYPE>::Type REAL;
        typedef ARRAY_TYPE ARRAY_CLASS;
        std::vector<int> gpus;

    public:
        gpBbSolver_lit() : gpSolver<ARRAY_TYPE>()
        {
            iterations_ = 10;
            tc_tolerance_ = (REAL)1e-6;
            non_negativity_constraint_ = false;
            dump_residual = false;
            threshold = REAL(1e-8);
        }

        virtual ~gpBbSolver_lit() {}

        virtual void set_gpus(std::vector<int> gpus_input)
        {
            this->gpus = gpus_input;
        }
        virtual boost::shared_ptr<ARRAY_TYPE> solve(ARRAY_TYPE *in)
        {
            if (this->encoding_operator_.get() == 0)
            {
                throw std::runtime_error("Error: gpBbSolver_lit::compute_rhs : no encoding operator is set");
            }

            // Get image space dimensions from the encoding operator
            //

            boost::shared_ptr<std::vector<size_t>> image_dims = this->encoding_operator_->get_domain_dimensions();
            if (image_dims->size() == 0)
            {
                throw std::runtime_error("Error: gpBbSolver_lit::compute_rhs : encoding operator has not set domain dimension");
            }

            if (this->gpus.size() > 1)
                cudaSetDevice(this->gpus[0]);

            ARRAY_TYPE *x = new ARRAY_TYPE;
            x->create(image_dims.get());

            if (this->gpus.size() > 1)
                cudaSetDevice(this->gpus[1]);

            ARRAY_TYPE x_old(image_dims.get());

            if (this->gpus.size() > 1)
                cudaSetDevice(this->gpus[0]);

            ARRAY_TYPE *g = new ARRAY_TYPE;
            g->create(image_dims.get());

            if (this->gpus.size() > 1)
                cudaSetDevice(this->gpus[1]);

            ARRAY_TYPE *g_old = new ARRAY_TYPE;
            g_old->create(image_dims.get());

            if (this->gpus.size() > 1)
                cudaSetDevice(this->gpus[0]);

            if (this->x0_.get())
            {
                *x = *(this->x0_.get());
            }
            else
            {
                clear(x);
            }

            ARRAY_TYPE encoding_space;
            REAL reg_res, data_res;
            ARRAY_TYPE in_save;
            if (this->output_mode_ >= solver<ARRAY_TYPE, ARRAY_TYPE>::OUTPUT_VERBOSE)
            {
                GDEBUG_STREAM("Iterating..." << std::endl);
            }
            for (int i = 0; i < iterations_; i++)
            {
                if ((i == 0) && (!this->x0_.get()))
                {
                    clear(g);
                    if (this->gpus.size() > 1)
                    {
                        GDEBUG_STREAM("SetupGPUs");
                        cudaSetDevice(this->gpus[0]);
                    }
                    g->create(image_dims.get());
                    this->encoding_operator_->mult_MH(in, g);
                    if (precond_.get())
                    {
                        precond_->apply(g, g);
                        precond_->apply(g, g);
                    }

                    *g *= -this->encoding_operator_->get_weight();
                    data_res = real(dot(in, in));

                    reg_res = REAL(0);

                    if (this->gpus.size() > 1)
                    {
                        cudaSetDevice(this->gpus[1]);
                        (*in) = nhlbi_toolbox::utils::set_device(in, this->gpus[1]);
                        //= in_save;
                        cudaSetDevice(this->gpus[0]);
                        // in_save.clear();
                    }
                }
                else
                {
                    if (this->gpus.size() > 1)
                    {
                        cudaSetDevice(this->gpus[1]);
                        (*in) = nhlbi_toolbox::utils::set_device(in, this->gpus[1]);
                        //= in_save;
                        cudaSetDevice(this->gpus[0]);
                        // in_save.clear();
                    }
                    GDEBUG_STREAM("this->gpus.size(): " << this->gpus.size() << " Def GPU" << this->gpus[0]);
                    if (this->gpus.size() > 1)
                    {
                        GDEBUG_STREAM("SetupGPUs 2: " << i << " Def GPU" << this->gpus[0]);
                        cudaSetDevice(this->gpus[0]);
                    }
                    if ((i == 0 && (this->x0_.get())) || (i == 1 && (!this->x0_.get())))
                    {
                        GDEBUG_STREAM("Created");
                        encoding_space.create(*in->get_dimensions());
                    }
                    this->encoding_operator_->mult_M(x, &encoding_space);
                    if (this->gpus.size() > 1)
                    {
                        cudaSetDevice(this->gpus[1]);

                        encoding_space = nhlbi_toolbox::utils::set_device(&encoding_space, this->gpus[1]);

                        axpy(REAL(-1), in, &encoding_space);

                        encoding_space = nhlbi_toolbox::utils::set_device(&encoding_space, this->gpus[0]);

                        cudaSetDevice(this->gpus[0]);
                    }
                    else
                        axpy(REAL(-1), in, &encoding_space);

                    cudaDeviceSynchronize();
                    data_res = real(dot(&encoding_space, &encoding_space));
                    this->encoding_operator_->mult_MH(&encoding_space, g);
                    if (precond_.get())
                    {
                        precond_->apply(g, g);
                        precond_->apply(g, g);
                    }
                    *g *= this->encoding_operator_->get_weight();
                }

                this->add_gradient(x, g); // Adds the gradient from all the regularization operators

                if (this->output_mode_ >= solver<ARRAY_TYPE, ARRAY_TYPE>::OUTPUT_VERBOSE)
                {
                    GDEBUG_STREAM("Data residual: " << data_res << std::endl);
                }

                if (non_negativity_constraint_)
                    solver_non_negativity_filter(x, g);
                ELEMENT_TYPE nabla;
                if (i == 0)
                {
                    cudaSetDevice(this->gpus[0]);
                    ARRAY_TYPE tmp_encoding;
                    if (this->gpus.size() > 1)
                    {
                        cudaSetDevice(this->gpus[1]);

                        encoding_space = nhlbi_toolbox::utils::set_device(&encoding_space, this->gpus[1]);

                        tmp_encoding = nhlbi_toolbox::utils::set_device(&(*in), this->gpus[0]);

                        cudaSetDevice(this->gpus[0]);
                    }
                    else
                        tmp_encoding = *in;


                    this->encoding_operator_->mult_M(g, &tmp_encoding);

                    if (this->x0_.get())
                    {
                        if (this->gpus.size() > 1)
                        {
                            cudaSetDevice(this->gpus[1]);

                            tmp_encoding = nhlbi_toolbox::utils::set_device(&tmp_encoding, this->gpus[1]);

                            nabla = dot(&encoding_space, &tmp_encoding) / dot(&tmp_encoding, &tmp_encoding);

                            encoding_space = nhlbi_toolbox::utils::set_device(&encoding_space, this->gpus[0]);

                            cudaSetDevice(this->gpus[0]);

                        }
                        else
                            nabla = dot(&encoding_space, &tmp_encoding) / dot(&tmp_encoding, &tmp_encoding);
                    }
                    else
                    {
                        if (this->gpus.size() > 1)
                        {
                            tmp_encoding = nhlbi_toolbox::utils::set_device(&tmp_encoding, this->gpus[1]);
                            cudaSetDevice(this->gpus[1]);
                        }
                        nabla = -dot(in, &tmp_encoding) / dot(&tmp_encoding, &tmp_encoding);
                        if (this->gpus.size() > 1)
                        {
                            cudaSetDevice(this->gpus[0]);
                        }
                    }
                    if (this->gpus.size() > 1)
                    {
                        cudaSetDevice(this->gpus[0]);

                        encoding_space = nhlbi_toolbox::utils::set_device(&encoding_space, this->gpus[0]);

                        cudaSetDevice(this->gpus[0]);
                    }
                }
                else
                {
                    if (this->gpus.size() > 1)
                    {
                        cudaSetDevice(this->gpus[1]);

                        auto y = nhlbi_toolbox::utils::set_device(x, this->gpus[1]);
                        auto yg = nhlbi_toolbox::utils::set_device(g, this->gpus[1]);
                        cudaDeviceSynchronize();
                        x_old -= y;
                        *g_old -= yg;
                        cudaSetDevice(this->gpus[0]);
                    }
                    else
                    {
                        x_old -= *x;
                        *g_old -= *g;
                    }

                    if (this->gpus.size() > 1)
                    {
                        cudaSetDevice(this->gpus[1]);
                        ELEMENT_TYPE xx = dot(&x_old, &x_old);
                        ELEMENT_TYPE gx = dot(g_old, &x_old);

                        ELEMENT_TYPE nabla1 = xx / gx;

                        /* This is the code that enables the adaptive step size.
                 REAL gg = dot(g_old,&x_old);
                 REAL nabla2 = gx/gg;
                 if ((nabla2/nabla1) < 0.5) nabla = nabla2;
                 else nabla = nabla1;*/
                        nabla = nabla1;
                        cudaSetDevice(this->gpus[0]);
                    }
                    else
                    {

                        ELEMENT_TYPE xx = dot(&x_old, &x_old);
                        ELEMENT_TYPE gx = dot(g_old, &x_old);

                        ELEMENT_TYPE nabla1 = xx / gx;

                        /* This is the code that enables the adaptive step size.
                 REAL gg = dot(g_old,&x_old);
                 REAL nabla2 = gx/gg;
                 if ((nabla2/nabla1) < 0.5) nabla = nabla2;
                 else nabla = nabla1;*/
                        nabla = nabla1;
                    }
                }

                ARRAY_TYPE *tmp;
                ARRAY_TYPE *xnew;
                tmp = g_old;
                if (this->gpus.size() > 1)
                {
                    auto yg = nhlbi_toolbox::utils::set_device(g, this->gpus[1]);
                    auto y = nhlbi_toolbox::utils::set_device(x, this->gpus[1]);

                    *g_old = yg;
                    x_old = *x;
                }
                else
                {
                    g_old = g;
                    x_old = *x;
                }
                REAL grad_norm;
                if (this->gpus.size() > 1)
                {
                    cudaSetDevice(this->gpus[1]);
                    grad_norm = nrm2(g_old);
                    cudaSetDevice(this->gpus[0]);
                }
                else
                    grad_norm = nrm2(g_old);

                if (this->output_mode_ >= solver<ARRAY_TYPE, ARRAY_TYPE>::OUTPUT_VERBOSE)
                {
                    GDEBUG_STREAM("Iteration " << i << ". Gradient norm: " << grad_norm << std::endl);
                }
                iteration_callback(x, i, data_res, reg_res);
                if (this->gpus.size() > 1)
                {
                    // cudaSetDevice(this->gpus[1]);
                    // ARRAY_TYPE *y = new ARRAY_TYPE;
                    // y->create(image_dims.get());

                    axpy(-nabla, g, x);
                    
                    if (prox_enabled){
                        GDEBUG_STREAM("I am here");
                        xnew =g;
                        this->add_prox(x, xnew);
                        *x = xnew;
                    }
                    *g = nhlbi_toolbox::utils::set_device(tmp, this->gpus[0]);
                    // *x = nhlbi_toolbox::utils::set_device(y, this->gpus[0]);
                    cudaSetDevice(this->gpus[0]);
                }
                else
                {
                    axpy(-nabla, g_old, x);
                    g = tmp;
                    if (prox_enabled){
                        GDEBUG_STREAM("I am here");
                        xnew=g_old;
                        this->add_prox(x, xnew);
                        *x = xnew;
                    }
                    
                }
                
                if (non_negativity_constraint_)
                    clamp_min(x, REAL(0));
                if (grad_norm < tc_tolerance_)
                    break;
            }
            delete g;
            delete g_old;
            if (this->gpus.size() > 1)
            {
                (*in) = nhlbi_toolbox::utils::set_device(in, this->gpus[0]);
            }
            return boost::shared_ptr<ARRAY_TYPE>(x);
        }

        // Set preconditioner
        //
        /*virtual void set_preconditioner( boost::shared_ptr< cgPreconditioner<ARRAY_TYPE> > precond ) {
          precond_ = precond;
          }*/

        // Set/get maximally allowed number of iterations
        //
        virtual void set_max_iterations(unsigned int iterations) { iterations_ = iterations; }
        virtual unsigned int get_max_iterations() { return iterations_; }

        // Set/get tolerance threshold for termination criterium
        //
        virtual void set_tc_tolerance(REAL tolerance) { tc_tolerance_ = tolerance; }
        virtual REAL get_tc_tolerance() { return tc_tolerance_; }

        virtual void set_non_negativity_constraint(bool non_negativity_constraint)
        {
            non_negativity_constraint_ = non_negativity_constraint;
        }

        virtual void set_dump_residual(bool dump_res)
        {
            dump_residual = dump_res;
        }
        // Set preconditioner
        //

        virtual void set_preconditioner(boost::shared_ptr<cgPreconditioner<ARRAY_TYPE>> precond)
        {
            precond_ = precond;
        }

        virtual void add_regularization_operator_prox(boost::shared_ptr< linearOperator<ARRAY_TYPE> > op, int L_norm ){
            prox_enabled =true;
            if (L_norm==1){
                GDEBUG_STREAM("Not implemented");
                //operators_prox.push_back(boost::shared_ptr<gpRegularizationOperator>(new l1GPRegularizationOperator(op)));
            }else{
                operators_prox.push_back(op);
            }
        }

    protected:
        typedef typename std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>>::iterator csIterator;
        typedef typename std::vector<std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>>>::iterator csGroupIterator;

        virtual void iteration_callback(ARRAY_TYPE *, int i, REAL, REAL){};

    protected:
        // Preconditioner
        // boost::shared_ptr< cgPreconditioner<ARRAY_TYPE> > precond_;
        // Maximum number of iterations

        virtual void add_prox(ARRAY_TYPE* x, ARRAY_TYPE* y){
            for (int i = 0; i < operators_prox.size(); i++){
                boost::shared_ptr<generalOperator<ARRAY_TYPE> > op = operators_prox[i];
                op->gradient(x,y,false);
            }
            }

        unsigned int iterations_;
        bool non_negativity_constraint_;
        REAL tc_tolerance_;
        REAL threshold;
        bool dump_residual;
        bool prox_enabled=false;
        // Preconditioner
        boost::shared_ptr<cgPreconditioner<ARRAY_TYPE>> precond_;
        std::vector< boost::shared_ptr< generalOperator<ARRAY_TYPE> > > operators_prox;
    };
}
