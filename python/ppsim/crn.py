"""
Module for expressing population protocols using CRN notation. Ideas and much code taken from
https://github.com/enricozb/python-crn.

The general syntax is

.. code-block:: python

    a, b, u = species('A B U')
    approx_majority = [
        a + b >> 2 * u,
        a + u >> 2 * a,
        b + u >> 2 * b,
    ]
    n = 10 ** 5
    init_config = {a: 0.51 * n, b: 0.49 * n}
    sim = Simulation(init_config=init_config, rule=approx_majority)

In other words, a list of reactions is treated by the ppsim library just like the other ways of specifying
population protocol transitions (the `rule` parameter in the constructor for 
:class:`ppsim.simulation.Simulation`, which also
accepts a dict or a Python function).

More examples given in https://github.com/UC-Davis-molecular-computing/ppsim/tree/main/examples
"""

from __future__ import annotations  # needed for forward references in type hints

from collections import defaultdict
import copy
from enum import Enum
from typing import Iterable, Any, TypeAlias
from dataclasses import dataclass
from xml.dom import minidom
import gpac as gp

CATALYST_NAME = "K"
WASTE_NAME = "W"

def species(sp: str | Iterable[str]) -> tuple[Specie, ...]:
    """
    Create a list of :class:`Specie` (Single species :class:`Expression`'s),
    or a single one.

    args:
        sp:
            An string or Iterable of strings representing the names of the species being created.
            If a single string, species names are interpreted as space-separated.

    Examples:

    .. code-block:: python

        w, x, y, z = species('W X Y Z')
        rxn = x + y >> z + w


    .. code-block:: python

        w, x, y, z = species(['W', 'X', 'Y', 'Z'])
        rxn = x + y >> z + w

    """
    species_names_list: list[str]
    if isinstance(sp, str):
        species_names_list = sp.split()
    else:
        species_names_list = [specie.strip() for specie in sp]

    # if len(species_list) == 1:
    #     return Specie(species_list[0])
    if len(species_names_list) != len(set(species_names_list)):
        raise ValueError(f'species_list {species_names_list} cannot contain duplicates.')

    return tuple(Specie(name) for name in species_names_list)


SpeciePair: TypeAlias = tuple['Specie', 'Specie']  # forward annotations don't seem to work here
Output: TypeAlias = SpeciePair | dict[SpeciePair, float]


def replace_reversible_rxns(rxns: Iterable[Reaction]) -> list[Reaction]:
    """
    Args:
        rxns: list of :class:`Reaction`'s

    Returns: list of :class:`Reaction`'s, where every reversible reaction in `rxns` has been replaced by
        two irreversible reactions, and all others have been left as they are
    """
    new_rxns: list[Reaction] = []
    for rxn in rxns:
        if not rxn.reversible:
            new_rxn = copy.deepcopy(rxn)
            new_rxns.append(new_rxn)
        else:
            forward_rxn = Reaction(reactants=rxn.reactants, products=rxn.products, k=rxn.rate_constant,
                                   rate_constant_units=rxn.rate_constant_units, reversible=False)
            reverse_rxn = Reaction(reactants=rxn.products, products=rxn.reactants,
                                   k=rxn.rate_constant_reverse,
                                   rate_constant_units=rxn.rate_constant_reverse_units, reversible=False)
            new_rxns.extend([forward_rxn, reverse_rxn])
    return new_rxns


# XXX: This algorithm currently uses the reactant *ordered* pair.
# We should think about the logic of that and see if it makes sense to collapse
# two reversed ordered pairs to a single unordered pair at this step,
# or whether that should be done explicitly by the user specifying transition_order='symmetric'.
def reactions_to_dict(reactions: Iterable[Reaction], n: int, volume: float) \
        -> tuple[dict[SpeciePair, Output], float]:
    """
    Returns dict representation of `reactions`, transforming unimolecular reactions to bimolecular,
    and converting rates to probabilities, also returning the max rate so the :class:`ppsim.simulation.Simulation` knows
    how to scale time.

    Args:
        reactions: list of :class:`Reaction`'s
        n: the population size, necessary for rate conversion
        volume: parameter as defined in Gillespie algorithm

    Returns:
        (transitions_dict, max_rate), where `transitions_dict` is the dict representation of the transitions,
        and `max_rate` is the maximum rate for any pair of reactants,
        i.e., if we have reactions (a + b >> c + d).k(2) and (a + b >> x + y).k(3),
        then the ordered pair (a,b) has rate 2+3 = 5
    """
    reactions = replace_reversible_rxns(reactions)

    # Make a copy of reactions because this conversion will mutate the reactions
    reactions = convert_unimolecular_to_bimolecular_and_flip_reactant_order(reactions, n, volume)

    # for each ordered pair of reactants, calculate sum of rate constants across all reactions with those reactants
    # get in stochastic units no matter how they're encoded in the reactions
    reactant_pair_rates: defaultdict[SpeciePair, float] = defaultdict(float)
    for reaction in reactions:
        if not reaction.is_bimolecular():
            raise ValueError(f'all reactions must have exactly two reactants, violated by {reaction}')
        if not reaction.num_products() == 2:
            raise ValueError(f'all reactions must have exactly two products, violated by {reaction}')
        reactants = reaction.reactants_if_bimolecular()
        reactant_pair_rates[reactants] += reaction.rate_constant_stochastic

    # divide all rate constants by the max rate (per reactant pair) to help turn them into probabilities
    max_rate = max(reactant_pair_rates.values())

    # add one randomized transition per reaction
    transitions: dict[SpeciePair, Output] = {}
    for reaction in reactions:
        prob = reaction.rate_constant_stochastic / max_rate
        reactants = reaction.reactants_if_bimolecular()
        products = reaction.products_if_exactly_two()
        if reactants not in transitions:
            # should we be worried about floating-point error here?
            # I hope not since dividing a float by itself should always result in 1.0.
            transitions[reactants] = products if prob == 1.0 else {products: prob}
        else:
            # if we calculated probabilities correctly above, if we assigned a dict entry to be non-randomized
            # (since prob == 1.0 above), we should not encounter another reaction with same reactants
            output = transitions[reactants]
            assert isinstance(output, dict)
            output[products] = prob

    # assert that each possible input for transitions has output probabilities summing to 1
    for reactants, outputs in transitions.items():
        if isinstance(outputs, dict):
            sum_probs = sum(prob for prob in outputs.values())
            assert sum_probs <= 1 + 2 ** -20, f'sum_probs exceeds 1: {sum_probs}'

    return transitions, max_rate


def convert_unimolecular_to_bimolecular_and_flip_reactant_order(reactions: Iterable[Reaction], n: int,
                                                                volume: float) -> list[Reaction]:
    """Process all reactions before being added to the dictionary.

    bimolecular reactions have their rates multiplied by the corrective factor (n-1) / (2 * volume).
    Bimolecular reactions with two different reactants are added twice, with their reactants in both orders.
    """

    # gather set of all species together
    all_species: set[Specie] = set()
    for reaction in reactions:
        all_species.update(reaction.get_species())

    converted_reactions = []
    for reaction in reactions:
        if reaction.num_reactants() != reaction.num_products():
            raise ValueError(
                f'each reaction must have same number of reactants and products, violated by {reaction}')
        if reaction.is_bimolecular():
            # Corrective factor to reaction rate
            reaction.rate_constant *= (n - 1) / (2 * volume)
            converted_reactions.append(reaction)
            # Add a flipped copy if the reaction has two different reactants
            reactants = reaction.reactants_if_bimolecular()
            if len(set(reactants)) > 1:
                flipped_reaction = copy.copy(reaction)
                flipped_reaction.reactants = Expression([reactants[1], reactants[0]])
                assert flipped_reaction.reactants_if_bimolecular != reactants
                converted_reactions.append(flipped_reaction)

        elif reaction.is_unimolecular():
            # for each unimolecular reaction R -->(k) P and each species S,
            # add a bimolecular reaction R+S -->(k) P+S
            reactant = reaction.reactant_if_unimolecular()
            product = reaction.product_if_unique()
            bimolecular_implementing_reactions = [(reactant + s >> product + s).k(reaction.rate_constant)
                                                  for s in all_species]

            converted_reactions.extend(bimolecular_implementing_reactions)
        else:
            raise ValueError(f'each reaction must have exactly one or two reactants, violated by {reaction}')
    return converted_reactions


@dataclass(frozen=True)
class Specie:
    name: str
    is_special_specie: bool = False

    def __add__(self, other: Specie | Expression) -> Expression:
        if isinstance(other, Expression):
            return other + Expression([self])
        elif isinstance(other, Specie):
            return Expression([self]) + Expression([other])

        raise NotImplementedError()

    __radd__ = __add__

    def __rshift__(self, other: Specie | Expression | None) -> Reaction:
        if other is not None:
            return Reaction(self, other)
        else:
            return Reaction(self, Expression([]))

    def __rrshift__(self, other: Specie | Expression | None) -> Reaction:
        if other is not None:
            return Reaction(other, self)
        else:
            return Reaction(Expression([]), self)

    def __or__(self, other: Specie | Expression) -> Reaction:
        return Reaction(self, other, reversible=True)

    def __mul__(self, other: int) -> Expression:
        if isinstance(other, int):
            return other * Expression([self])
        else:
            raise NotImplementedError()

    def __rmul__(self, other: int) -> Expression:
        if isinstance(other, int):
            return other * Expression([self])
        else:
            raise NotImplementedError()

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Specie):
            return NotImplemented
        return self.name < other.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Specie):
            return NotImplemented
        return self.name == other.name and self.is_special_specie == other.is_special_specie

    __req__ = __eq__


@dataclass(frozen=True)
class Expression:
    """
    Class used for very basic symbolic manipulation of left/right hand
    side of stoichiometric equations. Not very user friendly; users should
    just use the `species` functions and manipulate those to get their
    reactions.
    """

    species: list[Specie]
    """ordered list of species in expression, e.g, A+A+B is [A,A,B]"""

    def __add__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            species_copy = list(self.species)
            species_copy.extend(other.species)
            return Expression(species_copy)
        elif isinstance(other, Specie):
            return self + Expression([other])
        else:
            raise NotImplementedError()

    def __rmul__(self, coeff: int) -> Expression:
        if isinstance(coeff, int):
            species_copy = []
            for _ in range(coeff):
                species_copy.extend(self.species)
            return Expression(species_copy)
        else:
            raise NotImplementedError()

    __mul__ = __rmul__

    def __rshift__(self, expr: Specie | Expression | None) -> Reaction:
        if expr is not None:
            return Reaction(self, expr)
        else:
            return Reaction(self, Expression([]))

    def __or__(self, other: Specie | Expression) -> Reaction:
        return Reaction(self, other, reversible=True)

    def __str__(self) -> str:
        return ' + '.join(s.name for s in self.species)

    def __len__(self) -> int:
        return len(self.species)

    def get_species(self) -> set[Specie]:
        """
        Returns the set of species in this expression, not their
        coefficients.
        """
        return set(self.species)


avogadro = 6.02214076e23


def concentration_to_count(concentration: float, volume: float) -> int:
    """

    Args:
        concentration: units of M (molar) = moles / liter
        volume: units of liter

    Returns:
        count of molecule with `concentration` in `volume`
    """
    return round(avogadro * concentration * volume)


class RateConstantUnits(Enum):
    stochastic = 'stochastic'
    """Units of L/s. Multiple by Avogadro's number to convert to mass-action units."""

    mass_action = 'mass_action'
    """Units of /M/s. Divide by Avogadro's number to convert to stochastic units."""


@dataclass
class Reaction:
    """
    Representation of a stoichiometric reaction using a pair of Expressions,
    one for the reactants and one for the products.
    """

    reactants: Expression
    """The left side of species in the reaction."""

    products: Expression
    """The right side of species in the reaction."""

    rate_constant: float = 1
    """Rate constant of forward reaction."""

    rate_constant_reverse: float = 1
    """Rate constant of reverse reaction (only used if :py:data:`Reaction.reversible` is true)."""

    rate_constant_units: RateConstantUnits = RateConstantUnits.stochastic
    """Units of forward rate constant."""

    rate_constant_reverse_units: RateConstantUnits = RateConstantUnits.stochastic
    """Units of reverse rate constant."""

    reversible: bool = False
    """Whether reaction is reversible, i.e. products &rarr; reactants is a reaction also."""

    def __init__(self, reactants: Specie | Expression, products: Specie | Expression,
                 k: float = 1, r: float = 1,
                 rate_constant_units: RateConstantUnits = RateConstantUnits.stochastic,
                 rate_constant_reverse_units: RateConstantUnits = RateConstantUnits.stochastic,
                 reversible: bool = False) -> None:
        """
        Args:
            reactants: left side of species in the reaction
            products: right side of species in the reaction
            k: Rate constant of forward reaction
            r: Rate constant of reverse reaction (only used if :py:data:`Reaction.reversible` is true
            rate_constant_units: Units of forward rate constant
            rate_constant_reverse_units: Units of reverse rate constant
            reversible: Whether reaction is reversible
        """
        if not (isinstance(reactants, Specie) or isinstance(reactants, Expression)):
            raise ValueError(
                "Attempted construction of reaction with type of reactants "
                f"as {type(reactants)}. Type of reactants must be Species "
                "or Expression")
        if not (isinstance(products, Specie) or isinstance(products, Expression)):
            raise ValueError(
                "Attempted construction of products with type of products "
                f"as {type(products)}. Type of products must be Species "
                "or Expression")

        if isinstance(reactants, Specie):
            reactants = Expression([reactants])
        if isinstance(products, Specie):
            products = Expression([products])

        self.reactants = reactants
        self.products = products
        self.rate_constant = float(k)
        self.rate_constant_reverse = float(r)
        self.rate_constant_units = rate_constant_units
        self.rate_constant_reverse_units = rate_constant_reverse_units
        self.reversible = reversible

    def is_unimolecular(self) -> bool:
        """
        Returns: true if there is one reactant
        """
        return self.num_reactants() == 1

    def is_bimolecular(self) -> bool:
        """
        Returns: true if there are two reactants
        """
        return self.num_reactants() == 2

    def symmetric(self) -> bool:
        """
        Returns: true if there are two reactants that are the same species
        """
        return self.num_reactants() == 2 and self.reactants.species[0] == self.reactants.species[1]

    def symmetric_products(self) -> bool:
        """
        Returns: true if there are two products that are the same species
        """
        return self.num_products() == 2 and self.products.species[0] == self.products.species[1]

    def num_reactants(self) -> int:
        """
        Returns: number of reactants
        """
        return len(self.reactants)

    def num_products(self) -> int:
        """
        Returns: number of products
        """
        return len(self.products)

    def is_conservative(self) -> bool:
        """
        Returns: true if number of reactants equals number of products
        """
        return self.num_reactants() == self.num_products()

    def reactant_if_unimolecular(self) -> Specie:
        """
        Returns: unique reactant if there is only one
        Raises: ValueError if there are multiple reactants
        """
        if self.is_unimolecular():
            return self.reactants.species[0]
        else:
            raise ValueError(f'reaction {self} is not unimolecular')

    def product_if_unique(self) -> Specie:
        """
        Returns: unique product if there is only one
        Raises: ValueError if there are multiple products
        """
        if self.num_products() == 1:
            return self.products.species[0]
        else:
            raise ValueError(f'reaction {self} does not have exactly one product')

    def reactants_if_bimolecular(self) -> tuple[Specie, Specie]:
        """
        Returns: pair of reactants if there are exactly two
        Raises: ValueError if there are not exactly two reactants
        """
        if self.is_bimolecular():
            return self.reactants.species[0], self.reactants.species[1]
        else:
            raise ValueError(f'reaction {self} is not bimolecular')

    def reactant_names_if_bimolecular(self) -> tuple[str, str]:
        """
        Returns: pair of reactant names if there are exactly two
        Raises: ValueError if there are not exactly two reactants
        """
        r1, r2 = self.reactants_if_bimolecular()
        return r1.name, r2.name

    def products_if_exactly_two(self) -> tuple[Specie, Specie]:
        """
        Returns: pair of products if there are exactly two
        Raises: ValueError if there are not exactly two products
        """
        if self.num_products() == 2:
            return self.products.species[0], self.products.species[1]
        else:
            raise ValueError(f'reaction {self} does not have exactly two products')

    def product_names_if_exactly_two(self) -> tuple[str, str]:
        """
        Returns: pair of product names if there are exactly two
        Raises: ValueError if there are not exactly two products
        """
        p1, p2 = self.products_if_exactly_two()
        return p1.name, p2.name

    def generativity(self) -> int:
        """
        Returns: number of products minus number of reactants
        """
        return self.num_products() - self.num_reactants()

    def __str__(self) -> str:
        rev_rate_str = f'({self.rate_constant_reverse})<' if self.reversible else ''
        return f"{self.reactants} {rev_rate_str}-->({self.rate_constant}) {self.products}"

    def __repr__(self) -> str:
        return (f"Reaction({repr(self.reactants)}, {repr(self.products)}, "
                f"{self.rate_constant})")

    @property
    def rate_constant_stochastic(self) -> float:
        """
        Returns: forward rate constant in stochastic units (converts from mass-action if necessary)
        """
        return self.rate_constant \
            if self.rate_constant_units == RateConstantUnits.stochastic \
            else self.rate_constant / avogadro

    @property
    def rate_constant_reverse_stochastic(self) -> float:
        """
        Returns: reverse rate constant in stochastic units (converts from mass-action if necessary)
        """
        return self.rate_constant_reverse \
            if self.rate_constant_reverse_units == RateConstantUnits.stochastic \
            else self.rate_constant_reverse / avogadro

    def k(self, coeff: float, units: RateConstantUnits = RateConstantUnits.stochastic) -> Reaction:
        """
        Changes the reaction coefficient to `coeff` and returns `self`.

        This is useful for including the rate constant during the construction
        of a reaction. For example

        .. code-block:: python

            x, y, z = species("X Y Z")
            rxns = [
                (x + y >> z).k(2.5),
                (z >> x).k(1.5),
                (z >> y).k(0.5)),
            ]

        args:
            coeff: float
                The new reaction coefficient
            units: float
                units of rate constant (default stochastic)
        """
        if self.is_unimolecular() and units == RateConstantUnits.mass_action:
            raise ValueError('cannot use mass-action rate constants on a unimolecular reaction')
        self.rate_constant_units = units
        self.rate_constant = coeff
        return self

    def r(self, coeff: float, units: RateConstantUnits = RateConstantUnits.stochastic) -> Reaction:
        """
        Changes the reverse reaction reaction rate constant to `coeff` and returns `self`.

        This is useful for including the rate constant during the construction
        of a reaction. For example

        .. code-block:: python

            x, y, z = species("X Y Z")
            rxns = [
                (x + y >> z).k(2.5),
                (z >> x).k(1.5),
                (z >> y).k(0.5)),
            ]

        args:
            coeff: float
                The new reverse reaction rate constant
            units: float
                units of rate constant (default stochastic)
        """
        if self.num_products() == 1 and units == RateConstantUnits.mass_action:
            raise ValueError('cannot use mass-action rate constants on a unimolecular reaction; '
                             'this reaction has only one product, so its reverse is unimolecular')
        if not self.reversible:
            raise ValueError('cannot set r on an irreversible reaction')
        self.rate_constant_reverse_units = units
        self.rate_constant_reverse = coeff
        return self

    def get_species(self) -> tuple[Specie]:
        """
        Return: 
            the species present in the products and reactants.
        """
        all_species = []
        all_species_set = set()
        for s in self.reactants.species + self.products.species:
            if s not in all_species_set:
                all_species.append(s)
                all_species_set.add(s)
        return tuple(all_species)

@dataclass
class CRN:
    """
    Representation of a CRN, a set of reactions.
    """

    reactions: list[Reaction]
    """The reactions comprising the CRN."""

    species: list[Specie]
    """The species in the CRN. Ordered by index that represents them during algorithm execution."""

    def __str__(self) -> str:
        return str([str(rxn) for rxn in self.reactions])


    def order(self) -> int:
        """
        Returns: the order of the CRN, the greatest number of reactants in any reaction
        """
        return max(rxn.num_reactants() for rxn in self.reactions)

    def generativity(self) -> int:
        """
        Returns: the generativity of the CRN, the greatest number of
        products minus reactants in any reaction
        """
        return max(rxn.generativity() for rxn in self.reactions)

    def exportable_reactions(self) -> list[tuple[list[int], list[int], float]]:
        """
        Returns: the reactions of the CRN formatted in a way that can be passed through pyo3.
        """
        reactions: list[tuple[list[int], list[int], float]] = []
        all_species: set[Specie] = set()
        for reaction in self.reactions:
            all_species.update(reaction.reactants.get_species())
            all_species.update(reaction.products.get_species())
        species_to_index: dict[Specie, int] = {}

        for i in range(len(self.species)):
            specie = self.species[i]
            species_to_index[specie] = i
            
        for reaction in self.reactions:
            reactants = list(map(lambda species: species_to_index[species], reaction.reactants.species))
            products = list(map(lambda species: species_to_index[species], reaction.products.species))
            reactions.append((reactants, products, reaction.rate_constant))
        return reactions

# For consistency, to ensure that the crn and the simulator see the reactants in the same order.
def extra_species():
    return [catalyst_specie(), waste_specie()]

def convert_to_uniform(crn:CRN, volume: float) -> CRN:
    """
    Convert a CRN to an equivalent uniform CRN.
    The new CRN will have two new species, K (catalyst) and W (waste).
    This function does NOT adjust rate constants for the count of K; this is done later.

    Args:
        crn: a CRN to be converted. Should not be an output of this function.
        volume: volume of the CRN.
    """
    # Special species are added by this function, so there shouldn't be any yet.
    for specie in crn.species:
        assert not specie.is_special_specie
    max_generativity = crn.generativity()
    max_order = crn.order()
    new_reactions:list[Reaction] = []
    K = catalyst_specie()
    W = waste_specie()
    new_species = crn.species + extra_species()
    for reaction in replace_reversible_rxns(crn.reactions):
        reaction_order = reaction.num_reactants()
        reaction_generativity = reaction.generativity()
        K_to_add = max_order - reaction_order
        W_to_add = max_generativity - reaction_generativity
        new_reactants = reaction.reactants + (K_to_add * K)
        new_products = reaction.products + (K_to_add * K) + (W_to_add * W)
        new_k = reaction.rate_constant / volume**(reaction_order - 1)
        new_reactions.append(Reaction(reactants=new_reactants, products=new_products, k=new_k))


    return CRN(reactions=new_reactions, species=new_species)

def catalyst_specie() -> Specie:
    return Specie(name=CATALYST_NAME,is_special_specie=True)

def waste_specie() -> Specie:
    return Specie(name=WASTE_NAME,is_special_specie=True)

# example of StochKit format:
'''
<Model>
   <Description>Epidemic</Description>
   <NumberOfReactions>1</NumberOfReactions>
   <NumberOfSpecies>2</NumberOfSpecies>
   <ParametersList>
     <Parameter>
       <Id>c1</Id>
       <Expression>1.0</Expression>
     </Parameter>
   </ParametersList>
   <ReactionsList>
     <Reaction>
       <Id>R2</Id>
       <Description> A+B -> 2B </Description>
       <Type>mass-action</Type>
       <Rate>c1</Rate>
       <Reactants>
           <SpeciesReference id="A" stoichiometry="1"/>
           <SpeciesReference id="B" stoichiometry="1"/>
       </Reactants>
       <Products>
           <SpeciesReference id="B" stoichiometry="2"/>
       </Products>
     </Reaction>
  </ReactionsList>
  <SpeciesList>
     <Species>
       <Id>A</Id>
       <Description>Species #1</Description>
       <InitialPopulation>10000</InitialPopulation>
     </Species>
     <Species>
       <Id>B</Id>
       <Description>Species #2</Description>
       <InitialPopulation>1</InitialPopulation>
     </Species>
  </SpeciesList>
</Model>
'''


def species_in_rxns(rxns: Iterable[Reaction]) -> list[Specie]:
    """
    Args:
        rxns: iterable of :class:`Reaction`'s

    Returns: list of species (without repetitions) in :class:`Reaction`'s in `rxns`
    """
    species_set: set[Specie] = set()
    species_list: list[Specie] = []
    for rxn in rxns:
        for sp in rxn.reactants.species + rxn.products.species:
            if sp not in species_set:
                species_set.add(sp)
                species_list.append(sp)
    return species_list

def gpac_format(rxns: Iterable[Reaction], init_config: dict[Specie, int]) -> tuple[list[gp.Reaction], dict[gp.Specie, int]]:
    """
    Create a gpac CRN in the form of equivalent initial configuration and list of gpac Reaction objects.

    These can be simulated using the Gillespie algorithm (with the rebop package as a backend)
    in gpac using gpac.rebop_crn_counts, and plotted using gpac.plot_crn_counts.

    Args:
        rxns: reactions to translate to gpac
        init_config: dict mapping each (ppsim) :class:`Specie` to its initial count

    Returns:
        A tuple of (reactions, config) essentially equivalent to the ppsim init_config and rxns
        but using the gpac package's versions of those data structures.
    """
    pp_species_set = set()
    for rxn in rxns:
        for sp in rxn.get_species():
            pp_species_set.add(sp)
    gp_init = {gp.Specie(name=sp.name): count for sp, count in init_config.items()}
    gp_rxns = []
    for rxn in rxns:
        pp_reactants = rxn.reactants.species
        pp_products = rxn.products.species
        gp_reactants = [gp.Specie(name=sp.name) for sp in pp_reactants]
        gp_products = [gp.Specie(name=sp.name) for sp in pp_products]
        gp_reactants_expr = gp.Expression(gp_reactants)
        gp_products_expr = gp.Expression(gp_products)
        gp_rxn = (gp_reactants_expr >> gp_products_expr).k(rxn.rate_constant)
        if rxn.reversible:
            gp_rxn.r(rxn.rate_constant_reverse)
        gp_rxns.append(gp_rxn)
    
    return gp_rxns, gp_init
    

def gillespy2_format(init_config: dict[Specie, int], rxns: Iterable[Reaction],
                     volume: float = 1.0) -> Any:
    """
    Create a gillespy2 Model object from a CRN description.

    Args:
        init_config: dict mapping each :class:`Specie` to its initial count
        rxns: reactions to translate to StochKit format
        volume: volume in liters

    Returns:
        An equivalent gillespy2 Model object
    """
    # requires package gillespy2 to be installed
    import gillespy2 # type: ignore

    rxns = replace_reversible_rxns(rxns)
    species_list = species_in_rxns(rxns)
    model = gillespy2.Model()

    init_config = defaultdict(int, init_config)

    gillespy2_species = {s: gillespy2.Species(name=s.name, initial_value=init_config[s]) for s in
                         species_list}
    model.add_species(list(gillespy2_species.values()))
    model.volume = volume
    rates = [gillespy2.Parameter(name='r' + str(i), expression=r.rate_constant) for i, r in enumerate(rxns)]
    model.add_parameter(rates)
    for rxn, rate in zip(rxns, rates):
        reactants = {gillespy2_species[s]: rxn.reactants.species.count(s) for s in
                     rxn.reactants.get_species()}
        # Divide rate by 2 in same-species bimolecular reaction because gillespy2 propensity would be x(x-1)
        if list(reactants.values()) == [2]:
            assert rate.expression is not None
            rate.expression = float(rate.expression) / 2
        products = {gillespy2_species[s]: rxn.products.species.count(s) for s in rxn.products.get_species()}
        model.add_reaction(gillespy2.Reaction(reactants=reactants, products=products, rate=rate))
    return model


def stochkit_format(rxns: Iterable[Reaction], init_config: dict[Specie, int], 
                    volume: float = 1.0, name: str = 'CRN') -> str:
    """

    Args:
        rxns: reactions to translate to StochKit format
        init_config: dict mapping each :class:`Specie` to its initial count
        volume: volume in liters
        name: name of the CRN

    Returns:
        string describing CRN in StochKit XML format
    """
    rxns = replace_reversible_rxns(rxns)
    species_list = species_in_rxns(rxns)

    root = minidom.Document()

    model = root.createElement('Model')
    root.appendChild(model)

    desc_node = root.createElement('Description')
    model.appendChild(desc_node)
    desc_text = root.createTextNode(name)
    desc_node.appendChild(desc_text)

    num_rxns_node = root.createElement('NumberOfReactions')
    model.appendChild(num_rxns_node)
    num_rxns_text = root.createTextNode(str(len(rxns)))
    num_rxns_node.appendChild(num_rxns_text)

    num_species_node = root.createElement('NumberOfSpecies')
    model.appendChild(num_species_node)
    num_species_text = root.createTextNode(str(len(species_list)))
    num_species_node.appendChild(num_species_text)

    species_node = root.createElement('SpeciesList')
    model.appendChild(species_node)
    for specie in species_list:
        specie_node = root.createElement('Species')
        species_node.appendChild(specie_node)

        id_node = root.createElement('Id')
        specie_node.appendChild(id_node)
        id_text = root.createTextNode(specie.name)
        id_node.appendChild(id_text)

        specie_description_node = root.createElement('Description')
        specie_node.appendChild(specie_description_node)
        specie_description_text = root.createTextNode(specie.name)
        specie_description_node.appendChild(specie_description_text)

        initial_population_node = root.createElement('InitialPopulation')
        specie_node.appendChild(initial_population_node)
        count = init_config.get(specie, 0)
        initial_population_text = root.createTextNode(str(count))
        initial_population_node.appendChild(initial_population_text)

    rxns_node = root.createElement('ReactionsList')
    model.appendChild(rxns_node)
    for idx, rxn in enumerate(rxns):
        rxn_node = root.createElement('Reaction')
        rxns_node.appendChild(rxn_node)

        id_node = root.createElement('Id')
        rxn_node.appendChild(id_node)
        id_text = root.createTextNode(f'R{idx}')
        id_node.appendChild(id_text)

        description_node = root.createElement('Description')
        rxn_node.appendChild(description_node)
        description_text = root.createTextNode(str(rxn))
        description_node.appendChild(description_text)

        type_node = root.createElement('Type')
        rxn_node.appendChild(type_node)
        type_text = root.createTextNode('mass-action')
        type_node.appendChild(type_text)

        rate_node = root.createElement('Rate')
        rxn_node.appendChild(rate_node)
        rate = rxn.rate_constant_stochastic
        if rxn.is_bimolecular():
            rate /= volume
        rate_text = root.createTextNode(f'{rate}')
        rate_node.appendChild(rate_text)

        # reactants
        reactants_node = root.createElement('Reactants')
        rxn_node.appendChild(reactants_node)
        for reactant in rxn.reactants.get_species():
            reactant_node = root.createElement('SpeciesReference')
            reactants_node.appendChild(reactant_node)
            reactant_node.setAttribute('id', reactant.name)
            reactant_node.setAttribute('stoichiometry', str(rxn.reactants.species.count(reactant)))

        # products
        products_node = root.createElement('Products')
        rxn_node.appendChild(products_node)
        for product in rxn.products.get_species():
            product_node = root.createElement('SpeciesReference')
            products_node.appendChild(product_node)
            product_node.setAttribute('id', product.name)
            product_node.setAttribute('stoichiometry', str(rxn.products.species.count(product)))

    stochkit_xml: str = root.toprettyxml(indent='  ')

    return stochkit_xml


def write_stochkit_file(filename: str, rxns: Iterable[Reaction], init_config: dict[Specie, int],
                        volume: float = 1.0, name: str = 'CRN') -> None:
    """
    Write stochkit file
    Args:
        filename: name of file to write
        rxns: reactions to translate to StochKit format
        init_config: dict mapping each :class:`Specie` to its initial count
        volume: volume in liters
        name: name of the CRN
    """
    xml = stochkit_format(rxns, init_config, volume, name)
    with open(filename, 'w') as f:
        f.write(xml)
